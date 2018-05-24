use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

use failure::Error;

use hal::{Backend, Device as HalDevice};
use hal::device::WaitFor;
use hal::queue::{QueueFamilyId, RawCommandQueue, RawSubmission};
use hal::window::{Backbuffer, Frame, FrameSync, Swapchain, SwapchainConfig, Extent2D, Surface};

#[cfg(feature = "gfx-backend-metal")]
use metal;

use factory::Factory;

pub trait Render<B: Backend, T> {
    fn render(
        &mut self,
        frame: Frame,
        acquire: &B::Semaphore,
        swapchain: &mut B::Swapchain,
        fences: &mut Vec<B::Fence>,
        families: &mut HashMap<QueueFamilyId, Vec<B::CommandQueue>>,
        factory: &mut Factory<B>,
        data: &mut T,
    ) -> usize;

    fn dispose(self, factory: &mut Factory<B>, data: &mut T) -> Backbuffer<B>;
}

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd)]
pub struct TargetId(u64);

pub struct Renderer<B: Backend, R> {
    autorelease: AutoreleasePool<B>,
    
    targets: HashMap<TargetId, Target<B, R>>,
    resources: Resources<B>,
    counter: u64,
}

impl<B, R> Renderer<B, R>
where
    B: Backend,
{
    /// Dispose of the renderer.
    pub fn dispose<T>(mut self, factory: &mut Factory<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        for (_, target) in self.targets {
            target.dispose(factory, &mut self.resources, data);
        }
        self.resources.dispose(factory);
    }

    /// Creates new render
    pub fn add_target(
        &mut self,
        mut surface: B::Surface,
        config: SwapchainConfig,
        factory: &mut Factory<B>,
    ) -> TargetId {
        self.counter += 1;
        let id = TargetId(self.counter);
        debug_assert!(self.targets.get(&id).is_none());

        let (swapchain, backbuffer) = factory.create_swapchain(&mut surface, config);
        let target = Target {
            surface: surface,
            swapchain,
            backbuffer: Some(backbuffer),
            render: None,
            jobs: Jobs::new(),
        };
        self.targets.insert(id, target);
        id
    }

    /// Remove render
    pub fn remove_target<T>(&mut self, id: TargetId, factory: &mut Factory<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        if let Some(target) = self.targets.remove(&id) {
            target.dispose(factory, &mut self.resources, data);
        }
    }

    /// Add graph to the render
    pub fn set_render<F, E, T>(&mut self, id: TargetId, factory: &mut Factory<B>, data: &mut T, render: F) -> Result<(), Error>
    where
        F: FnOnce(Backbuffer<B>, Extent2D, &mut Factory<B>, &mut T) -> Result<R, E>,
        E: Into<Error>,
        R: Render<B, T>,
    {
        let ref mut target = *self.targets
            .get_mut(&id)
            .ok_or(format_err!("No target with id {:#?}", id))?;

        target.set_render(id, factory, &mut self.resources, data, render)
    }

    /// Create new render system providing it with general queue group and surfaces to draw onto
    pub fn new(families: HashMap<QueueFamilyId, Vec<B::CommandQueue>>) -> Self
    where
        R: Send + Sync,
    {
        fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<Self>();

        Renderer {
            autorelease: AutoreleasePool::new(),
            targets: HashMap::new(),
            counter: 0,
            resources: Resources {
                families,
                fences: Vec::new(),
                semaphores: Vec::new(),
            },
        }
    }

    pub fn run<T>(&mut self, factory: &mut Factory<B>, data: &mut T)
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
        if let Some(earliest) = self.targets
            .values()
            .filter_map(|target| target.jobs.earliest())
            .min() {

            unsafe {
                // cleanup after finished jobs.
                factory.advance(earliest);
            }
        }

        self.autorelease.reset();
    }

    fn poll_uploads(&mut self, factory: &mut Factory<B>)
    where
        B: Backend,
    {
        let ref mut families = self.resources.families;
        for (cbuf, fid) in factory.uploads() {
            let family = families.get_mut(&fid).unwrap();

            unsafe {
                family[0].submit_raw(
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
    surface: B::Surface,
    swapchain: B::Swapchain,
    backbuffer: Option<Backbuffer<B>>,
    render: Option<R>,
    jobs: Jobs<B>,
}

impl<B, R> Target<B, R>
where
    B: Backend,
{
    fn set_render<F, E, T>(&mut self, id: TargetId, factory: &mut Factory<B>, resources: &mut Resources<B>, data: &mut T, render: F) -> Result<(), Error>
    where
        F: FnOnce(Backbuffer<B>, Extent2D, &mut Factory<B>, &mut T) -> Result<R, E>,
        E: Into<Error>,
        R: Render<B, T>,
    {
        self.shutdown(factory, resources, data);
        let backbuffer = self.backbuffer.take().unwrap();
        let extent = self.surface.kind().extent().into();
        let render = render(backbuffer, extent, factory, data)
            .map_err(|err| err.into().context(format!("Failed to build `Render` for target: {:?}", id)))?;
        self.render = Some(render);
        Ok(())
    }

    fn shutdown<T>(&mut self, factory: &mut Factory<B>, resources: &mut Resources<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        // Wait for jobs to finish.
        self.jobs.wait(None, |payload| {
            if !factory.wait_for_fences(&payload.fences, WaitFor::All, !0) {
                panic!("Device lost or something");
            }
            factory.reset_fences(&payload.fences);
            resources.fences.extend(payload.fences);
            resources.semaphores.push(payload.acquire);
        });

        if let Some(backbuffer) = self.render.take().map(|render| render.dispose(factory, data)) {
            debug_assert!(self.backbuffer.is_none());
            self.backbuffer = Some(backbuffer);
        } else {
            debug_assert!(self.backbuffer.is_some());
        }
    }

    fn run<T>(&mut self, factory: &mut Factory<B>, resources: &mut Resources<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        let render = if let Some(ref mut render) = self.render {
            render
        } else {
            return
        };

        // Get fresh semaphore.
        let acquire = resources
            .semaphores
            .pop()
            .unwrap_or_else(|| factory.create_semaphore());

        // Start frame acquisition.
        let frame = self.swapchain.acquire_frame(FrameSync::Semaphore(&acquire));
        let index = factory.current();

        // Clean fences
        let mut fences = Vec::new();

        // Wait for earlier jobs until one associated with next frame.
        self.jobs.wait(Some(frame.clone()), |payload| {
            // Wait for job to finish.
            if !factory.wait_for_fences(&payload.fences, WaitFor::All, !0) {
                panic!("Device lost or something");
            }

            // Reclaim fences and acquisition semaphore
            fences.extend(payload.fences);
            resources.semaphores.push(payload.acquire);
        });

        // Enqueue frame.
        let job = self.jobs.push(frame.clone(), index);

        // Record and submit commands to draw frame.
        let fences_used = render.render(
            frame.clone(),
            &acquire,
            &mut self.swapchain,
            &mut fences,
            &mut resources.families,
            factory,
            data,
        );

        if fences_used < fences.len() {
            factory.reset_fences(&fences[fences_used..]);
            for fence in fences.drain(fences_used ..) {
                factory.destroy_fence(fence);
            }
        }

        // Save job resources.
        job.payload = Some(Payload {
            fences,
            acquire,
        });
    }

    fn dispose<T>(mut self, factory: &mut Factory<B>, resources: &mut Resources<B>, data: &mut T) -> Backbuffer<B>
    where
        R: Render<B, T>,
    {
        self.shutdown(factory, resources, data);
        self.backbuffer.take().unwrap()
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

    fn push(
        &mut self,
        frame: Frame,
        index: u64,
    ) -> &mut Job<B> {
        while frame.id() >= self.jobs.len() {
            self.jobs.push(Job {
                payload: None,
            });
        }
        self.queue.push_back((frame.clone(), index));
        &mut self.jobs[frame.id()]
    }

    fn wait<W>(&mut self, frame: Option<Frame>, mut waiting: W)
    where
        W: FnMut(Payload<B>),
    {
        while let Some((f, _)) = self.queue.pop_front() {
            waiting(self.jobs[f.id()].payload.take().unwrap());
            if frame.as_ref().map_or(false, |j| j.id() == f.id()) {
                break;
            }
        }
    }
}

struct Job<B: Backend> {
    payload: Option<Payload<B>>,
}

struct Payload<B: Backend> {
    acquire: B::Semaphore,
    fences: Vec<B::Fence>,
}

struct Resources<B: Backend> {
    families: HashMap<QueueFamilyId, Vec<B::CommandQueue>>,
    fences: Vec<B::Fence>,
    semaphores: Vec<B::Semaphore>,
}

impl<B> Resources<B>
where
    B: Backend,
{
    fn dispose(self, factory: &mut Factory<B>) {
        for semaphore in self.semaphores {
            factory.destroy_semaphore(semaphore);
        }
        for fence in self.fences {
            factory.destroy_fence(fence);
        }
    }
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
