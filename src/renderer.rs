
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

use hal::{Backend, Device as HalDevice};
use hal::command::{Rect, Viewport};
use hal::image::Kind;
use hal::pool::{CommandPool, CommandPoolCreateFlags};
use hal::queue::{General, QueueGroup, CommandQueue, RawCommandQueue, RawSubmission, Supports};
use hal::window::{Backbuffer, FrameSync, Surface, Swapchain, SwapchainConfig, Frame as SurfaceFrame};

#[cfg(feature = "gfx-backend-metal")]
use metal;

use Error;
use factory::Factory;

pub trait Render<B: Backend, T> {
    fn render<C>(&mut self, &mut CommandQueue<B, C>, &mut CommandPool<B, C>, &Backbuffer<B>, SurfaceFrame,
    &B::Semaphore, &B::Semaphore, Viewport, &B::Fence, &mut Factory<B>, data: &mut T)
    where
        C: Supports<General>;
}


#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub struct TargetId(u64);

pub struct Renderer<B: Backend, R> {
    autorelease: AutoreleasePool<B>,
    queues_usage: Vec<usize>,
    targets: HashMap<TargetId, Target<B, R>>,
    resources: Resources<B>,
    counter: u64,
}

impl<B, R> Renderer<B, R>
where
    B: Backend,
{
    /// Creates new render
    pub fn add_target(
        &mut self,
        mut surface: B::Surface,
        config: SwapchainConfig,
        device: &B::Device,
    ) -> TargetId {
        self.counter += 1;
        let id = TargetId(self.counter);
        debug_assert!(self.targets.get(&id).is_none());

        let queue = self.queues_usage
            .iter()
            .enumerate()
            .min_by_key(|&(_, u)| u)
            .map(|(i, _)| i)
            .expect("There are some queues");
        self.queues_usage[queue] += 1;
        let (swapchain, backbuffer) = device.create_swapchain(&mut surface, config);
        let target = Target {
            queue,
            surface,
            swapchain,
            backbuffer,
            active: None,
            renders: Vec::new(),
            frames: VecDeque::new(),
            jobs: Vec::new(),
        };
        self.targets.insert(id, target);
        id
    }

    /// Remove render
    pub fn remove_target(&mut self, _id: TargetId) {
        unimplemented!()
    }

    /// Add graph to the render
    pub fn add_render(
        &mut self,
        id: TargetId,
        render: R,
    ) -> Result<(), Error> {
        let ref mut target = *self.targets
            .get_mut(&id)
            .ok_or(format!("No render with id {:#?}", id))?;
        target.renders.push(render);
        Ok(())
    }

    /// Create new render system providing it with general queue group and surfaces to draw onto
    pub fn new(group: QueueGroup<B, General>) -> Self
    where
        R: Send + Sync,
    {
        fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<Self>();

        Renderer {
            autorelease: AutoreleasePool::new(),
            queues_usage: vec![0; group.queues.len()],
            targets: HashMap::new(),
            counter: 0,
            resources: Resources {
                group,
                pools: Vec::new(),
                fences: Vec::new(),
                semaphores: Vec::new(),
            },
        }
    }

    pub fn run<T>(&mut self, data: &mut T, factory: &mut Factory<B>)
    where
        B: Backend,
        R: Render<B, T>,
    {
        self.poll_uploads(factory);

        // Run targets
        for target in self.targets.values_mut() {
            target.run(factory, &mut self.resources, data);
        }

        // walk over frames and find earliest
        let earliest = self.targets
            .values()
            .filter_map(|target| target.frames.front())
            .map(|f| f.started)
            .min()
            .unwrap();

        unsafe {
            // cleanup after finished jobs.
            factory.advance(earliest);
        }

        self.autorelease.reset();
    }

    fn poll_uploads(&mut self, factory: &mut Factory<B>)
    where
        B: Backend,
    {
        if let Some((cbuf, _)) = factory.uploads() {
            if self.resources.group.queues.len() > 1 {
                unimplemented!("Upload in multiqueue environment is not supported yet");
            }
            unsafe {
                self.resources.group.queues[0].as_mut().submit_raw(
                    RawSubmission {
                        cmd_buffers: Some(cbuf),
                        wait_semaphores: &[],
                        signal_semaphores: &[],
                    },
                    None,
                );
            }
        }
    }
}

struct Target<B: Backend, R> {
    queue: usize,
    surface: B::Surface,
    swapchain: B::Swapchain,
    backbuffer: Backbuffer<B>,
    active: Option<usize>,
    renders: Vec<R>,
    frames: VecDeque<Frame>,
    jobs: Vec<Job<B>>,
}

impl<B, R> Target<B, R>
where
    B: Backend,
{
    fn run<T>(&mut self, factory: &mut Factory<B>, resources: &mut Resources<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        if let Some(active) = self.active {
            // Get fresh semaphore.
            let acquire = resources.semaphores
                .pop()
                .unwrap_or_else(|| factory.create_semaphore());

            // Start frame acquisition.
            let surface_frame = self.swapchain.acquire_frame(FrameSync::Semaphore(&acquire));
            let frame = Frame {
                index: surface_frame.id(),
                started: factory.current(),
            };

            // Grow job vector.
            while frame.index >= self.jobs.len() {
                self.jobs.push(Job {
                    release: resources.semaphores
                        .pop()
                        .unwrap_or_else(|| factory.create_semaphore()),
                    payload: None,
                });
            }

            // Pop earliest jobs ...
            while let Some(f) = self.frames.pop_front() {
                // Get the job.
                let ref mut job = self.jobs[f.index];

                if let Some(Payload {
                    fence,
                    mut pool,
                    acquire,
                    ..
                }) = job.payload.take()
                {
                    // Wait for job to finish.
                    if !factory.wait_for_fence(&fence, !0) {
                        panic!("Device lost or something");
                    }
                    // reset fence and pool
                    factory.reset_fence(&fence);
                    pool.reset();

                    // Reclaim fence, pool and acquisition semaphore
                    resources.fences.push(fence);
                    resources.pools.push(pool);
                    resources.semaphores.push(acquire);
                }

                // ... until the job associated with current frame
                if f.index == frame.index {
                    break;
                }
            }

            let ref mut job = self.jobs[frame.index];

            let fence = resources.fences
                .pop()
                .unwrap_or_else(|| factory.create_fence(false));
            let mut pool = resources.pools.pop().unwrap_or_else(|| {
                factory.create_command_pool_typed(&resources.group, CommandPoolCreateFlags::TRANSIENT, 1)
            });

            // Get all required resources.
            let ref mut render = self.renders[active];
            let ref mut queue = resources.group.queues[self.queue];

            // Record and submit commands to draw frame.
            render.render(
                queue,
                &mut pool,
                &self.backbuffer,
                surface_frame,
                &acquire,
                &job.release,
                viewport(self.surface.kind()),
                &fence,
                factory,
                data,
            );

            // Setup presenting.
            queue.present(Some(&mut self.swapchain), Some(&job.release));

            // Save job resources.
            job.payload = Some(Payload {
                fence,
                acquire,
                pool,
            });

            // Enqueue frame.
            self.frames.push_back(frame);
        } else if !self.jobs.is_empty() {
            // Target wants to stop processing.
            // Wait for associated queue to become idle.
            resources.group.queues[self.queue]
                .wait_idle()
                .expect("Device lost or something");

            // Get all jobs
            for Job { release, payload } in self.jobs.drain(..) {
                if let Some(Payload {
                    fence,
                    mut pool,
                    acquire,
                    ..
                }) = payload
                {
                    // reset fence and pool
                    factory.reset_fence(&fence);
                    pool.reset();

                    // Reclaim fence, pool and semaphores
                    resources.fences.push(fence);
                    resources.pools.push(pool);
                    resources.semaphores.push(acquire);
                    resources.semaphores.push(release);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Frame {
    index: usize,
    started: u64,
}

struct Job<B: Backend> {
    release: B::Semaphore,
    payload: Option<Payload<B>>,
}

struct Payload<B: Backend> {
    acquire: B::Semaphore,
    fence: B::Fence,
    pool: CommandPool<B, General>,
}

fn viewport(kind: Kind) -> Viewport {
    match kind {
        Kind::D2(w, h, _) | Kind::D2Array(w, h, _, _) => Viewport {
            rect: Rect { x: 0, y: 0, w, h },
            depth: 0.0..1.0,
        },
        _ => panic!("Unsupported surface kind"),
    }
}


struct Resources<B: Backend> {
    group: QueueGroup<B, General>,
    pools: Vec<CommandPool<B, General>>,
    fences: Vec<B::Fence>,
    semaphores: Vec<B::Semaphore>,
}

struct AutoreleasePool<B> {
    #[cfg(feature = "gfx-backend-metal")]
    autorelease: Option<metal::AutoreleasePool>,
    _pd: PhantomData<*mut B>,
}

#[cfg(feature = "gfx-backend-metal")]
impl<B: 'static> AutoreleasePool<B> {
    #[inline(always)]
    fn new() -> Self {
        use std::any::TypeId;
        AutoreleasePool {
            autorelease: {
                if TypeId::of::<B>() == TypeId::of::<metal::Backend>() {
                    Some(unsafe { metal::AutoreleasePool::new() })
                } else {
                    None
                }
            },
            _pd: PhantomData,
        }
    }

    #[inline(always)]
    fn reset(&mut self) {
        if TypeId::of::<B>() == TypeId::of::<metal::Backend>() {
            unsafe {
                self.autorelease.as_mut().unwrap().reset();
            }
        }
    }
}

#[cfg(not(feature = "gfx-backend-metal"))]
impl<B: 'static> AutoreleasePool<B> {
    #[inline(always)]
    fn new() -> Self {
        AutoreleasePool { _pd: PhantomData }
    }

    #[inline(always)]
    fn reset(&mut self) {}
}

unsafe impl<B> Send for AutoreleasePool<B> {}
unsafe impl<B> Sync for AutoreleasePool<B> {}
