use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

use failure::Error;

use hal::{Backend, Device as HalDevice};
use hal::pool::{CommandPool, CommandPoolCreateFlags};
use hal::queue::{CommandQueue, General, QueueGroup, RawCommandQueue, RawSubmission, Supports};
use hal::window::{Backbuffer, Frame, FrameSync, Swapchain, SwapchainConfig};

#[cfg(feature = "gfx-backend-metal")]
use metal;

use factory::Factory;

pub trait Render<B: Backend, T> {
    fn render<C>(
        &mut self,
        queue: &mut CommandQueue<B, C>,
        pool: &mut CommandPool<B, C>,
        backbuffer: &Backbuffer<B>,
        frame: Frame,
        acquire: &B::Semaphore,
        release: &B::Semaphore,
        fence: &B::Fence,
        factory: &mut Factory<B>,
        data: &mut T,
    ) where
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
            render: None,
            jobs: Jobs::new(),
        };
        self.targets.insert(id, target);
        id
    }

    /// Remove render
    pub fn remove_target(&mut self, _id: TargetId) {
        unimplemented!()
    }

    /// Add graph to the render
    pub fn set_render(&mut self, id: TargetId, render: R) -> Result<Option<R>, Error> {
        use std::mem::replace;
 
        let ref mut target = *self.targets
            .get_mut(&id)
            .ok_or(format_err!("No render with id {:#?}", id))?;
        Ok(replace(&mut target.render, Some(render)))
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

        // walk over job_queue and find earliest
        let earliest = self.targets
            .values()
            .filter_map(|target| target.jobs.earliest())
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

struct Jobs<B: Backend> {
    queue: VecDeque<(Frame, u64)>,
    jobs: Vec<Job<B>>,
}

impl<B> Jobs<B>
where
    B: Backend,
{
    fn new() -> Self {
        Jobs {
            queue: VecDeque::new(),
            jobs: Vec::new(),
        }
    }

    fn earliest(&self) -> Option<u64> {
        self.queue.front().map(|&(_, index)| index)
    }

    fn enqueue(
        &mut self,
        frame: Frame,
        index: u64,
        factory: &mut Factory<B>,
        resources: &mut Resources<B>,
    ) -> &mut Job<B> {
        while frame.id() >= self.jobs.len() {
            self.jobs.push(Job {
                release: resources
                    .semaphores
                    .pop()
                    .unwrap_or_else(|| factory.create_semaphore()),
                payload: None,
            });
        }
        self.queue.push_back((frame.clone(), index));
        &mut self.jobs[frame.id()]
    }

    fn wait<F>(&mut self, frame: Option<Frame>, mut f: F)
    where
        F: FnMut(Payload<B>),
    {
        while let Some((j, _)) = self.queue.pop_front() {
            f(self.jobs[j.id()].payload.take().unwrap());
            if frame.as_ref().map_or(false, |f| f.id() == j.id()) {
                break;
            }
        }
    }

    fn clean<F>(&mut self, mut f: F)
    where
        F: FnMut(Option<Payload<B>>, B::Semaphore),
    {
        for Job { payload, release } in self.jobs.drain(..) {
            f(payload, release);
        }
    }
}

struct Target<B: Backend, R> {
    queue: usize,
    surface: B::Surface,
    swapchain: B::Swapchain,
    backbuffer: Backbuffer<B>,
    render: Option<R>,
    jobs: Jobs<B>,
}

impl<B, R> Target<B, R>
where
    B: Backend,
{
    fn clean(&mut self, factory: &mut Factory<B>, resources: &mut Resources<B>) {
        // Target wants to stop processing.
        // Wait for associated queue to become idle.
        resources.group.queues[self.queue]
            .wait_idle()
            .expect("Device lost or something");

        // Cleanup all jobs.
        self.jobs.clean(|payload, release| {
            payload.map(|mut payload| {
                // reset fence and pool
                factory.reset_fence(&payload.fence);
                payload.pool.reset();

                // Reclaim fence, pool and semaphores
                resources.fences.push(payload.fence);
                resources.pools.push(payload.pool);
                resources.semaphores.push(payload.acquire);
            });
            resources.semaphores.push(release);
        });
    }

    fn run<T>(&mut self, factory: &mut Factory<B>, resources: &mut Resources<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        if let Some(ref mut render) = self.render {
            // Get fresh semaphore.
            let acquire = resources
                .semaphores
                .pop()
                .unwrap_or_else(|| factory.create_semaphore());

            // Start frame acquisition.
            let frame = self.swapchain.acquire_frame(FrameSync::Semaphore(&acquire));
            let index = factory.current();

            // Wait for earliest jobs ...
            self.jobs.wait(Some(frame.clone()), |mut payload| {
                // Wait for job to finish.
                if !factory.wait_for_fence(&payload.fence, !0) {
                    panic!("Device lost or something");
                }
                // reset fence and pool
                factory.reset_fence(&payload.fence);
                payload.pool.reset();

                // Reclaim fence, pool and acquisition semaphore
                resources.fences.push(payload.fence);
                resources.pools.push(payload.pool);
                resources.semaphores.push(payload.acquire);
            });

            // Enqueue frame.
            let job = self.jobs.enqueue(frame.clone(), index, factory, resources);

            let fence = resources
                .fences
                .pop()
                .unwrap_or_else(|| factory.create_fence(false));
            let mut pool = resources.pools.pop().unwrap_or_else(|| {
                factory.create_command_pool_typed(
                    &resources.group,
                    CommandPoolCreateFlags::TRANSIENT,
                    1,
                )
            });

            // Get all required resources.
            let ref mut queue = resources.group.queues[self.queue];

            // Record and submit commands to draw frame.
            render.render(
                queue,
                &mut pool,
                &self.backbuffer,
                frame.clone(),
                &acquire,
                &job.release,
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
        } else {
            self.clean(factory, resources);
        }
    }

    fn dispose(mut self, factory: &mut Factory<B>, resources: &mut Resources<B>) {
        self.clean(factory, resources);
    }
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
        use std::any::TypeId;
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
