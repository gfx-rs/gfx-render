use std::collections::HashMap;
use std::marker::PhantomData;

use failure::Error;

use hal::{Backend, Device as HalDevice};
use hal::device::WaitFor;
use hal::queue::{QueueFamilyId, RawCommandQueue, RawSubmission};
use hal::window::{Backbuffer};

use winit::Window;

use backend::BackendEx;

#[cfg(feature = "gfx-backend-metal")]
use metal;

use factory::Factory;

pub trait Render<B: Backend, T> {
    fn run(
        &mut self,
        fences: &mut Vec<B::Fence>,
        queues: &mut HashMap<QueueFamilyId, Vec<B::CommandQueue>>,
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
    families: Families<B>,
    counter: u64,
}

impl<B, R> Renderer<B, R>
where
    B: BackendEx,
{
    /// Dispose of the renderer.
    pub fn dispose<T>(mut self, factory: &mut Factory<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        for (_, target) in self.targets {
            target.dispose(factory, data);
        }
        self.families.dispose(factory);
    }

    /// Creates new render target
    pub fn add_target(
        &mut self,
        window: &Window,
        factory: &mut Factory<B>,
    ) -> TargetId {
        self.counter += 1;
        let id = TargetId(self.counter);
        debug_assert!(self.targets.get(&id).is_none());

        let surface = factory.create_surface(window);

        let target = Target {
            surface,
            render: None,
            fences: Vec::new(),
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
            target.dispose(factory, data);
        }
    }

    /// Add graph to the render
    pub fn set_render<F, E, T>(&mut self, id: TargetId, factory: &mut Factory<B>, data: &mut T, render: F) -> Result<(), Error>
    where
        F: FnOnce(&mut B::Surface, &[B::QueueFamily], &mut Factory<B>, &mut T) -> Result<R, E>,
        E: Into<Error>,
        R: Render<B, T>,
    {
        let ref mut target = *self.targets
            .get_mut(&id)
            .ok_or(format_err!("No target with id {:#?}", id))?;

        target.wait(factory);
        target.render.take().map(|render| render.dispose(factory, data));
        target.render = Some(render(&mut target.surface, &self.families.families, factory, data).map_err(|err| err.into().context("Failed to build render"))?);


        Ok(())
    }

    /// Create new render system providing it with general queue group and surfaces to draw onto
    pub fn new(queues: HashMap<QueueFamilyId, Vec<B::CommandQueue>>, families: Vec<B::QueueFamily>,) -> Self
    where
        R: Send + Sync,
    {
        fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<Self>();

        Renderer {
            autorelease: AutoreleasePool::new(),
            targets: HashMap::new(),
            counter: 0,
            families: Families {
                queues,
                families,
            },
        }
    }

    #[inline]
    pub fn run<T>(&mut self, factory: &mut Factory<B>, data: &mut T)
    where
        B: Backend,
        R: Render<B, T>,
    {
        profile!("Renderer::run");
        self.poll_uploads(factory);

        // Run targets
        for target in self.targets.values_mut() {
            target.run(&mut self.families, factory, data);
        }

        unsafe {
            // cleanup after finished jobs.
            factory.advance();
        }

        self.autorelease.reset();
    }

    #[inline]
    fn poll_uploads(&mut self, factory: &mut Factory<B>)
    where
        B: Backend,
    {
        profile!("Renderer::poll_uploads");

        let ref mut queues = self.families.queues;
        for (cbuf, fid) in factory.uploads() {
            profile!("command buffer");

            let family = queues.get_mut(&fid).unwrap();

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
    render: Option<R>,
    fences: Vec<B::Fence>,
}

impl<B, R> Target<B, R>
where
    B: Backend,
{
    #[inline]
    fn wait(&mut self, factory: &mut Factory<B>) {
        profile!("Target::wait");
        if !self.fences.is_empty() && !factory.wait_for_fences(&self.fences, WaitFor::All, !0) {
            panic!("Device lost or something");
        }
    }

    #[inline]
    fn run<T>(&mut self, families: &mut Families<B>, factory: &mut Factory<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        profile!("Target::run");
        self.wait(factory);

        if let Some(ref mut render) = self.render {
            render.run(
                &mut self.fences,
                &mut families.queues,
                factory,
                data,
            );
        };
    }

    fn dispose<T>(mut self, factory: &mut Factory<B>, data: &mut T)
    where
        R: Render<B, T>,
    {
        self.wait(factory);
        unimplemented!()
    }
}

struct Families<B: Backend> {
    queues: HashMap<QueueFamilyId, Vec<B::CommandQueue>>,
    families: Vec<B::QueueFamily>,
}

impl<B> Families<B>
where
    B: Backend,
{
    fn dispose(self, factory: &mut Factory<B>) {
        unimplemented!()
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
        profile!("Autorelease");
        use std::any::TypeId;
        if TypeId::of::<B>() == TypeId::of::<metal::Backend>() {
            unsafe {
                profile!("Autorelease::reset");
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
