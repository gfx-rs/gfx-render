//! This module provide convenient `Factory` type that can be used for several tasks:
//! 1. Creating new `Buffer`s and `Image`s.
//! 2. Destroying `Buffer`s and `Image`s with additional safety.
//! 3. Uploading data to `Buffer`s and `Image`s.
//! 4. Creating `Surface`s and fetching their capabilities.
//! 5. Fetching `Features` and `Limits` of the GPU.
//! All this packet into sendable and shareable `Factory` type.
//!

use std::any::Any;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut, Range};

use hal::{Backend, Instance, Surface};
use hal::buffer::{Access as BufferAccess, Usage as BufferUsage};
use hal::format::Format;
use hal::image::{Access as ImageAccess, Extent, Kind, Layout, Level, Offset, StorageFlags,
                 SubresourceLayers, Tiling, Usage as ImageUsage};
use hal::memory::Properties;
use hal::queue;
use hal::window::SurfaceCapabilities;

use mem::{Block, Factory as FactoryTrait, SmartAllocator, SmartBlock, Type};

use winit::Window;

use Error;
use backend::BackendEx;
use escape::{Escape, Terminal};
use reclamation::ReclamationQueue;
use upload::Upload;

pub use mem::Item as RelevantItem;

/// Wrapper around raw gpu resource like `B::Buffer` or `B::Image`
/// It will send raw resource back to the `Factory` if dropped.
/// Destroying it manually with `Factory::destroy_*` is better performance-wise.
#[derive(Debug)]
pub struct Item<I, B> {
    inner: Escape<RelevantItem<I, B>>,
}

impl<I, B> Item<I, B> {
    /// Get raw gpu resource.
    pub fn raw(&self) -> &I {
        self.inner.raw()
    }

    /// Get memory block to which resource is bound.
    pub fn block(&self) -> &B {
        self.inner.block()
    }

    /// Unwrap from inner `Escape` wrapper.
    /// Returned item must be disposed manually.
    /// It may panic or behave unpredictably when dropped.
    pub fn into_inner(self) -> RelevantItem<I, B> {
        Escape::into_inner(self.inner)
    }
}

impl<I, B> Borrow<I> for Item<I, B> {
    fn borrow(&self) -> &I {
        (&*self.inner).borrow()
    }
}

impl<I, B> BorrowMut<I> for Item<I, B> {
    fn borrow_mut(&mut self) -> &mut I {
        (&mut *self.inner).borrow_mut()
    }
}

impl<I, B> Block for Item<I, B>
where
    I: Debug + Send + Sync,
    B: Block,
{
    type Memory = B::Memory;
    fn memory(&self) -> &Self::Memory {
        self.inner.memory()
    }
    fn range(&self) -> Range<u64> {
        self.inner.range()
    }
}

/// Buffer type `Factory` creates
pub type Buffer<B: Backend> = Item<B::Buffer, SmartBlock<B::Memory>>;

/// Image type `Factory` creates
pub type Image<B: Backend> = Item<B::Image, SmartBlock<B::Memory>>;

/// `Factory` is a central type that wraps GPU device and responsible for:
/// 1. Creating new `Buffer`s and `Image`s.
/// 2. Destroying `Buffer`s and `Image`s with additional safety.
/// 3. Uploading data to `Buffer`s and `Image`s.
/// 4. Creating `Surface`s and fetching their capabilities.
/// 5. Fetching `Features` and `Limits` of the GPU.
///
pub struct Factory<B: Backend> {
    instance: Box<Instance<Backend = B>>,
    physical: B::PhysicalDevice,
    device: B::Device,
    allocator: SmartAllocator<B>,
    reclamation: ReclamationQueue<AnyItem<B>>,
    current: u64,
    upload: Upload<B>,
    buffers: Terminal<RelevantBuffer<B>>,
    images: Terminal<RelevantImage<B>>,
}

impl<B> Factory<B>
where
    B: Backend,
{
    /// Create new `Buffer`. Factory will allocate buffer from memory which has all requested properties and supports all requested usages.
    ///
    /// # Parameters
    /// `size`          - size of buffer. Returned buffer _may_ be larger but never smaller.
    /// `properties`    - memory properties required for buffer.
    /// `usage`         - how buffer is supposed to be used. Caller must specify all usages and never use buffer otherwise.
    ///
    pub fn create_buffer(
        &mut self,
        size: u64,
        properties: Properties,
        usage: BufferUsage,
    ) -> Result<Buffer<B>, Error> {
        let buffer: RelevantBuffer<B> = self.allocator
            .create_buffer(
                self.device.borrow(),
                (Type::General, properties),
                size,
                usage,
            )
            .map_err(|err| Error::with_chain(err, "Failed to create buffer"))?;
        Ok(Item {
            inner: self.buffers.escape(buffer),
        })
    }

    /// Create new `Image`. Factory will allocate buffer from memory which has all requested properties and supports all requested usages.
    ///
    /// # Parameters
    ///
    /// `kind`          - image dimensions.
    /// `level`         - number of mim-levels.
    /// `format`        - format of the image.
    /// `properties`    - memory properties required for buffer.
    /// `usage`         - how buffer is supposed to be used. Caller must specify all usages and never use buffer otherwise.
    ///
    pub fn create_image(
        &mut self,
        kind: Kind,
        level: Level,
        format: Format,
        tiling: Tiling,
        properties: Properties,
        usage: ImageUsage,
        storage_flags: StorageFlags,
    ) -> Result<Image<B>, Error> {
        let image = self.allocator
            .create_image(
                self.device.borrow(),
                (Type::General, properties),
                kind,
                level,
                format,
                tiling,
                usage,
                storage_flags,
            )
            .map_err(|err| Error::with_chain(err, "Failed to create image"))?;
        Ok(Item {
            inner: self.images.escape(image),
        })
    }

    /// Destroy `Buffer`.
    /// Factory will destroy this buffer after all commands referencing this buffer will complete.
    pub fn destroy_buffer(&mut self, buffer: Buffer<B>) {
        self.reclamation
            .push(self.current, AnyItem::Buffer(buffer.into_inner()));
    }

    /// Destroy `Image`
    /// Factory will destroy this buffer after all commands referencing this image will complete.
    pub fn destroy_image(&mut self, image: Image<B>) {
        self.reclamation
            .push(self.current, AnyItem::Image(image.into_inner()));
    }

    /// Destroy `RelevantBuffer`
    /// Factory will destroy this buffer after all commands referencing this buffer will complete.
    pub fn destroy_relevant_buffer(&mut self, buffer: RelevantBuffer<B>) {
        self.reclamation.push(self.current, AnyItem::Buffer(buffer));
    }

    /// Destroy `RelevantImage`
    /// Factory will destroy this buffer after all commands referencing this image will complete.
    pub fn destroy_relevant_image(&mut self, image: RelevantImage<B>) {
        self.reclamation.push(self.current, AnyItem::Image(image));
    }

    /// Upload data to the buffer.
    /// Factory will try to use most appropriate way to write data to the buffer.
    /// For cpu-visible buffers it will write via memory mapping.
    /// If size of the `data` is bigger than `staging_threshold` then it will perform staging.
    /// Otherwise it will write through command buffer directly.
    ///
    /// # Parameters
    /// `buffer`    - where to upload data. It must be created with at least one of `TRANSFER_DST` usage or `CPU_VISIBLE` property.
    /// `offset`    - write data to the buffer starting from this byte.
    /// `data`      - data to upload.
    ///
    pub fn upload_buffer(
        &mut self,
        access: BufferAccess,
        buffer: &mut Buffer<B>,
        offset: u64,
        data: &[u8],
    ) -> Result<(), Error> {
        let ref device = self.device;
        let ref mut allocator = self.allocator;
        if let Some(staging) =
            self.upload
                .upload_buffer(device, allocator, &mut *buffer.inner, access, offset, data)?
        {
            self.reclamation
                .push(self.current, AnyItem::Buffer(staging));
        }
        Ok(())
    }

    /// Upload data to the image.
    /// Factory will use staging buffer to write data to the image.
    ///
    /// # Parameters
    ///
    /// `image`     - where to upload. It must be created with `TRANSFER_DST` usage.
    /// `layout`    - layout in which `Image` will be after uploading.
    /// `layers`    - specific image subresources of the image used for the destination image data.
    /// `offset`    - offsets in texels of the sub-region of the destination image data.
    /// `extent`    - size in texels of the sub-region of the destination image data.
    /// `data`      - data containing texels in image's format.
    pub fn upload_image(
        &mut self,
        image: &mut Image<B>,
        layout: Layout,
        access: ImageAccess,
        layers: SubresourceLayers,
        offset: Offset,
        extent: Extent,
        data: &[u8],
    ) -> Result<(), Error> {
        let ref device = self.device;
        let ref mut allocator = self.allocator;
        let staging = self.upload.upload_image(
            device,
            allocator,
            &mut *image.inner,
            access,
            layout,
            layers,
            offset,
            extent,
            data,
        )?;
        self.reclamation
            .push(self.current, AnyItem::Buffer(staging));
        Ok(())
    }

    /// Create new `Surface`.
    ///
    /// # Parameters
    ///
    /// `window`    - window handler which will be represented by new surface.
    pub fn create_surface(&mut self, window: &Window) -> B::Surface
    where
        B: BackendEx,
    {
        B::create_surface(
            Any::downcast_ref::<B::Instance>(&self.instance).unwrap(),
            window,
        )
    }

    /// Get capabilities and formats for surface.
    /// If formats are `None` then `Surface` has no preferences for formats.
    /// Otherwise `Swapchain` can be created only with one of formats returned by this function.
    ///
    /// # Parameters
    ///
    /// `surface`   - surface object which capabilities and supported formats are retrieved.
    ///
    pub fn capabilities_and_formats(
        &self,
        surface: &B::Surface,
    ) -> (SurfaceCapabilities, Option<Vec<Format>>) {
        surface.capabilities_and_formats(&self.physical)
    }

    /// Construct `Factory` from its parts.
    pub fn new(
        instance: B::Instance,
        physical: B::PhysicalDevice,
        device: B::Device,
        allocator: SmartAllocator<B>,
        staging_threshold: usize,
        upload_family: queue::QueueFamilyId,
    ) -> Self
    where
        B: BackendEx,
    {
        Factory {
            instance: Box::new(instance),
            physical: physical.into(),
            device: device.into(),
            allocator,
            reclamation: ReclamationQueue::new(),
            current: 0,
            upload: Upload::new(staging_threshold, upload_family),
            buffers: Terminal::new(),
            images: Terminal::new(),
        }
    }

    /// Borrow both `Device` and `SmartAllocator` from the `Factory`.
    pub fn device_and_allocator(&mut self) -> (&B::Device, &mut SmartAllocator<B>) {
        (self.device.borrow(), &mut self.allocator)
    }

    /// Fetch command buffer with uploads recorded.
    pub(crate) fn uploads(&mut self) -> Option<(&mut B::CommandBuffer, queue::QueueFamilyId)> {
        self.upload.uploads(self.current)
    }

    /// `RenderSystem` call this to know with which frame index recorded commands are associated.
    pub(crate) fn current(&mut self) -> u64 {
        self.current
    }

    /// `RenderSystem` call this with least frame index with which ongoing job is associated.
    /// Hence all resources released before this index can be destroyed.
    pub(crate) unsafe fn advance(&mut self, ongoing: u64) {
        debug_assert!(ongoing <= self.current);
        for buffer in self.buffers.drain() {
            self.reclamation.push(self.current, AnyItem::Buffer(buffer));
        }
        for image in self.images.drain() {
            self.reclamation.push(self.current, AnyItem::Image(image));
        }
        let ref device = self.device;
        let ref mut allocator = self.allocator;
        self.reclamation.clear(ongoing, |item| {
            item.destroy(device, allocator);
        });
        self.upload.clear(ongoing);
        self.current += 1;
    }
}

impl<B> Deref for Factory<B>
where
    B: Backend,
{
    type Target = B::Device;
    fn deref(&self) -> &B::Device {
        &self.device
    }
}

impl<B> DerefMut for Factory<B>
where
    B: Backend,
{
    fn deref_mut(&mut self) -> &mut B::Device {
        &mut self.device
    }
}

pub(crate) type RelevantBuffer<B: Backend> = RelevantItem<B::Buffer, SmartBlock<B::Memory>>;
pub(crate) type RelevantImage<B: Backend> = RelevantItem<B::Image, SmartBlock<B::Memory>>;

#[derive(Debug)]
enum AnyItem<B: Backend> {
    Buffer(RelevantBuffer<B>),
    Image(RelevantImage<B>),
}

impl<B> AnyItem<B>
where
    B: Backend,
{
    pub fn destroy(self, device: &B::Device, allocator: &mut SmartAllocator<B>) {
        match self {
            AnyItem::Buffer(buffer) => {
                allocator.destroy_buffer(device, buffer);
            }
            AnyItem::Image(image) => {
                allocator.destroy_image(device, image);
            }
        }
    }
}

#[test]
#[allow(dead_code)]
fn factory_send_sync() {
    fn is_send_sync<T: Send + Sync>() {}
    fn for_any_backend<B: Backend>() {
        is_send_sync::<Factory<B>>();
    }
}
