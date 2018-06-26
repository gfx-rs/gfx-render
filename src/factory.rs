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
use std::mem::ManuallyDrop;
use std::ops::Range;

use hal::{
    buffer, device::{BindError, FramebufferError, OutOfMemory, ShaderError, WaitFor},
    error::HostExecutionError, format::{Format, Swizzle}, image, mapping::{self, Reader, Writer},
    memory::{Properties, Requirements}, pass::{Attachment, SubpassDependency, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags}, pso, query::QueryType,
    queue::{QueueFamilyId, QueueGroup}, range::RangeArg, window, AdapterInfo, Backend, Device,
    Features, Limits, MemoryTypeId, PhysicalDevice, Surface,
};

use mem::{Block, MemoryAllocator, MemoryError, SmartAllocator, SmartBlock, Type};

use winit::Window;

use backend::BackendEx;
use escape::{Escape, Terminal};
use reclamation::ReclamationQueue;
use upload::{self, Upload};

/// Item that must be destructed manually
#[derive(Debug)]
pub struct RelevantItem<T, B> {
    raw: T,
    block: B,
}

impl<T, B> Borrow<T> for RelevantItem<T, B> {
    fn borrow(&self) -> &T {
        &self.raw
    }
}

impl<T, B> BorrowMut<T> for RelevantItem<T, B> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.raw
    }
}

impl<T, B> Block for RelevantItem<T, B>
where
    T: Debug + Send + Sync,
    B: Block,
{
    type Memory = B::Memory;
    fn memory(&self) -> &Self::Memory {
        self.block.memory()
    }
    fn range(&self) -> Range<u64> {
        self.block.range()
    }
}

/// Buffer that must be destructed manually
pub type RelevantBuffer<B> =
    RelevantItem<<B as Backend>::Buffer, SmartBlock<<B as Backend>::Memory>>;

/// Image that must be destructed manually
pub type RelevantImage<B> = RelevantItem<<B as Backend>::Image, SmartBlock<<B as Backend>::Memory>>;

/// Wrapper around raw gpu resource like `B::Buffer` or `B::Image`
/// It will send raw resource back to the `Factory` if dropped.
/// Destroying it manually with `Factory::destroy_*` is better performance-wise.
#[derive(Debug)]
pub struct Item<T, B> {
    inner: Escape<RelevantItem<T, B>>,
}

impl<T, B> Item<T, B> {
    /// Get raw gpu resource.
    pub fn raw(&self) -> &T {
        &self.inner.raw
    }

    /// Get memory block to which resource is bound.
    pub fn block(&self) -> &B {
        &self.inner.block
    }

    /// Unwrap from inner `Escape` wrapper.
    /// Returned item must be disposed manually.
    /// It may panic or behave unpredictably when dropped.
    pub fn into_inner(self) -> RelevantItem<T, B> {
        Escape::into_inner(self.inner)
    }
}

impl<T, B> Borrow<T> for Item<T, B> {
    fn borrow(&self) -> &T {
        &self.inner.raw
    }
}

impl<T, B> BorrowMut<T> for Item<T, B> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.inner.raw
    }
}

impl<T, B> Block for Item<T, B>
where
    T: Debug + Send + Sync,
    B: Block,
{
    type Memory = B::Memory;
    fn memory(&self) -> &Self::Memory {
        self.inner.block.memory()
    }
    fn range(&self) -> Range<u64> {
        self.inner.block.range()
    }
}

/// Buffer type `Factory` creates
pub type Buffer<B> = Item<<B as Backend>::Buffer, SmartBlock<<B as Backend>::Memory>>;

/// Image type `Factory` creates
pub type Image<B> = Item<<B as Backend>::Image, SmartBlock<<B as Backend>::Memory>>;

/// Errors this factory can produce.
#[derive(Clone, Debug, Fail)]
pub enum FactoryError {
    /// Buffer creation error.
    #[fail(display = "Failed to create buffer")]
    BufferCreationError(#[cause] BufferCreationError),

    /// Image creation error.
    #[fail(display = "Failed to create image")]
    ImageCreationError(#[cause] ImageCreationError),

    /// Uploading error.
    #[fail(display = "Failed to upload data")]
    UploadError(#[cause] upload::Error),
}

impl From<BufferCreationError> for FactoryError {
    fn from(error: BufferCreationError) -> Self {
        FactoryError::BufferCreationError(error)
    }
}

impl From<ImageCreationError> for FactoryError {
    fn from(error: ImageCreationError) -> Self {
        FactoryError::ImageCreationError(error)
    }
}

impl From<upload::Error> for FactoryError {
    fn from(error: upload::Error) -> Self {
        FactoryError::UploadError(error)
    }
}

/// Error occurred during buffer creation.
#[derive(Clone, Debug, Fail)]
pub enum BufferCreationError {
    /// Memory error reported from allocator.
    #[fail(display = "Failed to allocate memory")]
    MemoryError(#[cause] MemoryError),

    /// Creating error reported by driver.
    #[fail(display = "Failed to create buffer object")]
    CreatingError(#[cause] buffer::CreationError),
}

impl From<MemoryError> for BufferCreationError {
    fn from(error: MemoryError) -> Self {
        BufferCreationError::MemoryError(error)
    }
}

impl From<buffer::CreationError> for BufferCreationError {
    fn from(error: buffer::CreationError) -> Self {
        BufferCreationError::CreatingError(error)
    }
}

/// Error occurred during image creation.
#[derive(Clone, Debug, Fail)]
pub enum ImageCreationError {
    /// Memory error reported from allocator.
    #[fail(display = "Failed to allocate memory")]
    MemoryError(#[cause] MemoryError),

    /// Creating error reported by driver.
    #[fail(display = "Failed to create image object")]
    CreatingError(#[cause] image::CreationError),
}

impl From<MemoryError> for ImageCreationError {
    fn from(error: MemoryError) -> Self {
        ImageCreationError::MemoryError(error)
    }
}

impl From<image::CreationError> for ImageCreationError {
    fn from(error: image::CreationError) -> Self {
        ImageCreationError::CreatingError(error)
    }
}

/// `Factory` is a central type that wraps GPU device and responsible for:
/// 1. Creating new `Buffer`s and `Image`s.
/// 2. Destroying `Buffer`s and `Image`s with additional safety.
/// 3. Uploading data to `Buffer`s and `Image`s.
/// 4. Creating `Surface`s and fetching their capabilities.
/// 5. Fetching `Features` and `Limits` of the GPU.
///
pub struct Factory<B: Backend<Device = D>, D: Device<B> = <B as Backend>::Device> {
    device: D,
    allocator: ManuallyDrop<SmartAllocator<B>>,
    reclamation: ReclamationQueue<AnyItem<B>>,
    current: u64,
    upload: ManuallyDrop<Upload<B>>,
    buffers: Terminal<RelevantBuffer<B>>,
    images: Terminal<RelevantImage<B>>,
    instance: Box<Any + Send + Sync>,
    physical_device: B::PhysicalDevice,
    info: AdapterInfo,
}

impl<B, D> Drop for Factory<B, D>
where
    B: Backend<Device = D>,
    D: Device<B>,
{
    fn drop(&mut self) {
        use std::ptr::read;

        self.current = u64::max_value() - 1;
        debug!("Dispose of `Factory`");
        unsafe {
            debug!("Dispose of uploader");
            read(&mut *self.upload).dispose(&self.device);

            debug!("Advance to the end of times");
            self.advance();

            debug!("Dispose of allocator");
            read(&mut *self.allocator)
                .dispose(&self.device)
                .expect("Allocator must be cleared");
        }
    }
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
        usage: buffer::Usage,
        properties: Properties,
    ) -> Result<Buffer<B>, BufferCreationError> {
        Ok(Item {
            inner: self.buffers.escape(create_relevant_buffer(
                &self.device,
                &mut self.allocator,
                size,
                usage,
                Type::General,
                properties,
            )?),
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
        kind: image::Kind,
        level: image::Level,
        format: Format,
        tiling: image::Tiling,
        storage_flags: image::StorageFlags,
        usage: image::Usage,
        properties: Properties,
    ) -> Result<Image<B>, ImageCreationError> {
        Ok(Item {
            inner: self.images.escape(create_relevant_image(
                &self.device,
                &mut self.allocator,
                kind,
                level,
                format,
                tiling,
                storage_flags,
                usage,
                Type::General,
                properties,
            )?),
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
        buffer: &mut Buffer<B>,
        family: QueueFamilyId,
        access: buffer::Access,
        offset: u64,
        data: &[u8],
    ) -> Result<(), upload::Error> {
        let staging = unsafe {
            // Correct properties provided.
            let ref device = self.device;
            let ref mut allocator = self.allocator;
            let properties = allocator.properties(&buffer.block());
            self.upload.upload_buffer(
                device,
                |alloc_type, properties, size, usage| {
                    let buffer = create_relevant_buffer(
                        device, allocator, size, usage, alloc_type, properties,
                    )?;
                    let properties = allocator.properties(&buffer.block);
                    Ok((buffer, properties))
                },
                &mut buffer.inner,
                properties,
                family,
                access,
                offset,
                data,
            )?
        };
        staging.map(|staging| {
            self.reclamation
                .push(self.current, AnyItem::Buffer(staging));
        });
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
        family: QueueFamilyId,
        layout: image::Layout,
        access: image::Access,
        layers: image::SubresourceLayers,
        offset: image::Offset,
        extent: image::Extent,
        data_width: u32,
        data_height: u32,
        data: &[u8],
    ) -> Result<(), upload::Error> {
        let staging = unsafe {
            // Correct properties provided.
            let ref device = self.device;
            let ref mut allocator = self.allocator;
            self.upload.upload_image(
                device,
                |alloc_type, properties, size, usage| {
                    let buffer = create_relevant_buffer(
                        device, allocator, size, usage, alloc_type, properties,
                    )?;
                    let properties = allocator.properties(&buffer.block);
                    Ok((buffer, properties))
                },
                &mut image.inner,
                family,
                access,
                layout,
                layers,
                offset,
                extent,
                data_width,
                data_height,
                data,
            )?
        };
        self.reclamation
            .push(self.current, AnyItem::Buffer(staging));
        Ok(())
    }

    /// Create new `Surface`.
    ///
    /// # Parameters
    ///
    /// `window`    - window handler which will be represented by new surface.
    pub(crate) fn create_surface(&mut self, window: &Window) -> B::Surface
    where
        B: BackendEx,
    {
        use std::any::TypeId;
        trace!("Factory<{:?}>::create_surface", TypeId::of::<B>());
        let surface = Any::downcast_ref::<B::Instance>(&*self.instance).unwrap();
        B::create_surface(surface, window)
    }

    /// Get capabilities and formats for surface.
    /// If formats are `None` then `Surface` has no preferences for formats.
    /// Otherwise `Swapchain` can be created only with one of formats returned by this function.
    ///
    /// # Parameters
    ///
    /// `surface`   - surface object which capabilities and supported formats are retrieved.
    ///
    pub fn compatibility(
        &self,
        surface: &B::Surface,
    ) -> (
        window::SurfaceCapabilities,
        Option<Vec<Format>>,
        Vec<window::PresentMode>,
    ) {
        surface.compatibility(&self.physical_device)
    }

    /// Borrow both `Device` and `SmartAllocator` from the `Factory`.
    pub fn device_and_allocator(&mut self) -> (&B::Device, &mut SmartAllocator<B>) {
        (self.device.borrow(), &mut self.allocator)
    }

    /// Get features supported by hardware.
    pub fn features(&self) -> Features {
        self.physical_device.features()
    }

    /// Get hardware specific limits.
    pub fn limits(&self) -> Limits {
        self.physical_device.limits()
    }

    /// Retrieve adapter info
    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.info
    }

    /// Construct `Factory` from parts.
    pub(crate) fn new(
        instance: B::Instance,
        info: AdapterInfo,
        physical_device: B::PhysicalDevice,
        device: B::Device,
        allocator: SmartAllocator<B>,
        staging_threshold: usize,
    ) -> Self
    where
        B: BackendEx,
    {
        Factory {
            instance: Box::new(instance),
            device: device.into(),
            allocator: ManuallyDrop::new(allocator),
            reclamation: ReclamationQueue::new(),
            current: 0,
            upload: ManuallyDrop::new(Upload::new(staging_threshold)),
            buffers: Terminal::new(),
            images: Terminal::new(),
            physical_device,
            info,
        }
    }

    /// Fetch command buffer with uploads recorded.
    pub(crate) fn uploads(
        &mut self,
    ) -> impl Iterator<Item = (&mut B::CommandBuffer, QueueFamilyId)> {
        self.upload.uploads(self.current)
    }

    /// Render system call this after waiting for last jobs.
    pub(crate) unsafe fn advance(&mut self) {
        for buffer in self.buffers.drain() {
            self.reclamation.push(self.current, AnyItem::Buffer(buffer));
        }
        for image in self.images.drain() {
            self.reclamation.push(self.current, AnyItem::Image(image));
        }
        let ref device = self.device;
        let ref mut allocator = self.allocator;
        self.reclamation.clear(self.current, |item| {
            item.destroy(device, allocator);
        });
        self.upload.clear(self.current);
        self.current += 1;
    }
}

impl<B, D> Borrow<D> for Factory<B, D>
where
    B: Backend<Device = D>,
    D: Device<B>,
{
    fn borrow(&self) -> &D {
        &self.device
    }
}

impl<B, D> BorrowMut<D> for Factory<B, D>
where
    B: Backend<Device = D>,
    D: Device<B>,
{
    fn borrow_mut(&mut self) -> &mut D {
        &mut self.device
    }
}

impl<B, D> Device<B> for Factory<B, D>
where
    B: Backend<Device = D>,
    D: Device<B>,
{
    #[inline]
    fn allocate_memory(
        &self,
        memory_type: MemoryTypeId,
        size: u64,
    ) -> Result<B::Memory, OutOfMemory> {
        self.device.allocate_memory(memory_type, size)
    }

    #[inline]
    fn free_memory(&self, memory: B::Memory) {
        self.device.free_memory(memory)
    }

    #[inline]
    fn create_command_pool(
        &self,
        family: QueueFamilyId,
        create_flags: CommandPoolCreateFlags,
    ) -> B::CommandPool {
        self.device.create_command_pool(family, create_flags)
    }

    #[inline]
    fn destroy_command_pool(&self, pool: B::CommandPool) {
        self.device.destroy_command_pool(pool)
    }

    #[inline]
    fn create_render_pass<'a, IA, IS, ID>(
        &self,
        attachments: IA,
        subpasses: IS,
        dependencies: ID,
    ) -> B::RenderPass
    where
        IA: IntoIterator,
        IA::Item: Borrow<Attachment>,
        IS: IntoIterator,
        IS::Item: Borrow<SubpassDesc<'a>>,
        ID: IntoIterator,
        ID::Item: Borrow<SubpassDependency>,
    {
        self.device
            .create_render_pass(attachments, subpasses, dependencies)
    }

    #[inline]
    fn destroy_render_pass(&self, rp: B::RenderPass) {
        self.device.destroy_render_pass(rp)
    }

    #[inline]
    fn create_pipeline_layout<IS, IR>(
        &self,
        set_layouts: IS,
        push_constant: IR,
    ) -> B::PipelineLayout
    where
        IS: IntoIterator,
        IS::Item: Borrow<B::DescriptorSetLayout>,
        IR: IntoIterator,
        IR::Item: Borrow<(pso::ShaderStageFlags, Range<u32>)>,
    {
        self.device
            .create_pipeline_layout(set_layouts, push_constant)
    }

    #[inline]
    fn destroy_pipeline_layout(&self, layout: B::PipelineLayout) {
        self.device.destroy_pipeline_layout(layout)
    }

    #[inline]
    fn destroy_graphics_pipeline(&self, pipeline: B::GraphicsPipeline) {
        self.device.destroy_graphics_pipeline(pipeline)
    }

    #[inline]
    fn destroy_compute_pipeline(&self, pipeline: B::ComputePipeline) {
        self.device.destroy_compute_pipeline(pipeline)
    }

    #[inline]
    fn create_framebuffer<I>(
        &self,
        pass: &B::RenderPass,
        attachments: I,
        extent: image::Extent,
    ) -> Result<B::Framebuffer, FramebufferError>
    where
        I: IntoIterator,
        I::Item: Borrow<B::ImageView>,
    {
        self.device.create_framebuffer(pass, attachments, extent)
    }

    #[inline]
    fn destroy_framebuffer(&self, buf: B::Framebuffer) {
        self.device.destroy_framebuffer(buf)
    }

    #[inline]
    fn create_shader_module(&self, spirv_data: &[u8]) -> Result<B::ShaderModule, ShaderError> {
        self.device.create_shader_module(spirv_data)
    }

    #[inline]
    fn destroy_shader_module(&self, shader: B::ShaderModule) {
        self.device.destroy_shader_module(shader)
    }

    #[inline]
    fn create_buffer(
        &self,
        size: u64,
        usage: buffer::Usage,
    ) -> Result<B::UnboundBuffer, buffer::CreationError> {
        self.device.create_buffer(size, usage)
    }

    #[inline]
    fn get_buffer_requirements(&self, buf: &B::UnboundBuffer) -> Requirements {
        self.device.get_buffer_requirements(buf)
    }

    #[inline]
    fn bind_buffer_memory(
        &self,
        memory: &B::Memory,
        offset: u64,
        buf: B::UnboundBuffer,
    ) -> Result<B::Buffer, BindError> {
        self.device.bind_buffer_memory(memory, offset, buf)
    }

    #[inline]
    fn destroy_buffer(&self, buf: B::Buffer) {
        self.device.destroy_buffer(buf)
    }

    #[inline]
    fn create_buffer_view<R: RangeArg<u64>>(
        &self,
        buf: &B::Buffer,
        fmt: Option<Format>,
        range: R,
    ) -> Result<B::BufferView, buffer::ViewCreationError> {
        self.device.create_buffer_view(buf, fmt, range)
    }

    #[inline]
    fn destroy_buffer_view(&self, view: B::BufferView) {
        self.device.destroy_buffer_view(view)
    }

    #[inline]
    fn create_image(
        &self,
        kind: image::Kind,
        mip_levels: image::Level,
        format: Format,
        tiling: image::Tiling,
        usage: image::Usage,
        storage_flags: image::StorageFlags,
    ) -> Result<B::UnboundImage, image::CreationError> {
        self.device
            .create_image(kind, mip_levels, format, tiling, usage, storage_flags)
    }

    #[inline]
    fn get_image_requirements(&self, image: &B::UnboundImage) -> Requirements {
        self.device.get_image_requirements(image)
    }

    #[inline]
    fn bind_image_memory(
        &self,
        memory: &B::Memory,
        offset: u64,
        image: B::UnboundImage,
    ) -> Result<B::Image, BindError> {
        self.device.bind_image_memory(memory, offset, image)
    }

    #[inline]
    fn destroy_image(&self, image: B::Image) {
        self.device.destroy_image(image)
    }

    #[inline]
    fn create_image_view(
        &self,
        image: &B::Image,
        view_kind: image::ViewKind,
        format: Format,
        swizzle: Swizzle,
        range: image::SubresourceRange,
    ) -> Result<B::ImageView, image::ViewError> {
        self.device
            .create_image_view(image, view_kind, format, swizzle, range)
    }

    #[inline]
    fn destroy_image_view(&self, view: B::ImageView) {
        self.device.destroy_image_view(view)
    }

    #[inline]
    fn create_sampler(&self, info: image::SamplerInfo) -> B::Sampler {
        self.device.create_sampler(info)
    }

    #[inline]
    fn destroy_sampler(&self, sampler: B::Sampler) {
        self.device.destroy_sampler(sampler)
    }

    #[inline]
    fn create_descriptor_pool<I>(&self, max_sets: usize, descriptor_ranges: I) -> B::DescriptorPool
    where
        I: IntoIterator,
        I::Item: Borrow<pso::DescriptorRangeDesc>,
    {
        self.device
            .create_descriptor_pool(max_sets, descriptor_ranges)
    }

    #[inline]
    fn destroy_descriptor_pool(&self, pool: B::DescriptorPool) {
        self.device.destroy_descriptor_pool(pool)
    }

    #[inline]
    fn create_descriptor_set_layout<I, J>(
        &self,
        bindings: I,
        immutable_samplers: J,
    ) -> B::DescriptorSetLayout
    where
        I: IntoIterator,
        I::Item: Borrow<pso::DescriptorSetLayoutBinding>,
        J: IntoIterator,
        J::Item: Borrow<B::Sampler>,
    {
        self.device
            .create_descriptor_set_layout(bindings, immutable_samplers)
    }

    #[inline]
    fn destroy_descriptor_set_layout(&self, layout: B::DescriptorSetLayout) {
        self.device.destroy_descriptor_set_layout(layout)
    }

    #[inline]
    fn write_descriptor_sets<'a, I, J>(&self, write_iter: I)
    where
        I: IntoIterator<Item = pso::DescriptorSetWrite<'a, B, J>>,
        J: IntoIterator,
        J::Item: Borrow<pso::Descriptor<'a, B>>,
    {
        self.device.write_descriptor_sets(write_iter)
    }

    #[inline]
    fn copy_descriptor_sets<'a, I>(&self, copy_iter: I)
    where
        I: IntoIterator,
        I::Item: Borrow<pso::DescriptorSetCopy<'a, B>>,
    {
        self.device.copy_descriptor_sets(copy_iter)
    }

    #[inline]
    fn map_memory<R>(&self, memory: &B::Memory, range: R) -> Result<*mut u8, mapping::Error>
    where
        R: RangeArg<u64>,
    {
        self.device.map_memory(memory, range)
    }

    #[inline]
    fn flush_mapped_memory_ranges<'a, I, R>(&self, ranges: I)
    where
        I: IntoIterator,
        I::Item: Borrow<(&'a B::Memory, R)>,
        R: RangeArg<u64>,
    {
        self.device.flush_mapped_memory_ranges(ranges)
    }

    #[inline]
    fn invalidate_mapped_memory_ranges<'a, I, R>(&self, ranges: I)
    where
        I: IntoIterator,
        I::Item: Borrow<(&'a B::Memory, R)>,
        R: RangeArg<u64>,
    {
        self.device.invalidate_mapped_memory_ranges(ranges)
    }

    #[inline]
    fn unmap_memory(&self, memory: &B::Memory) {
        self.device.unmap_memory(memory)
    }

    #[inline]
    fn create_semaphore(&self) -> B::Semaphore {
        self.device.create_semaphore()
    }

    #[inline]
    fn destroy_semaphore(&self, semaphore: B::Semaphore) {
        self.device.destroy_semaphore(semaphore)
    }

    #[inline]
    fn create_fence(&self, signaled: bool) -> B::Fence {
        self.device.create_fence(signaled)
    }

    #[inline]
    fn get_fence_status(&self, fence: &B::Fence) -> bool {
        self.device.get_fence_status(fence)
    }

    #[inline]
    fn destroy_fence(&self, fence: B::Fence) {
        self.device.destroy_fence(fence)
    }

    #[inline]
    fn create_query_pool(&self, ty: QueryType, count: u32) -> B::QueryPool {
        self.device.create_query_pool(ty, count)
    }

    #[inline]
    fn destroy_query_pool(&self, pool: B::QueryPool) {
        self.device.destroy_query_pool(pool)
    }

    #[inline]
    fn create_swapchain(
        &self,
        surface: &mut B::Surface,
        config: window::SwapchainConfig,
        old_swapchain: Option<B::Swapchain>,
        extent: &window::Extent2D,
    ) -> (B::Swapchain, window::Backbuffer<B>) {
        self.device
            .create_swapchain(surface, config, old_swapchain, extent)
    }

    #[inline]
    fn destroy_swapchain(&self, swapchain: B::Swapchain) {
        self.device.destroy_swapchain(swapchain)
    }

    #[inline]
    fn wait_idle(&self) -> Result<(), HostExecutionError> {
        self.device.wait_idle()
    }

    #[inline]
    fn create_command_pool_typed<C>(
        &self,
        group: &QueueGroup<B, C>,
        flags: CommandPoolCreateFlags,
        max_buffers: usize,
    ) -> CommandPool<B, C> {
        self.device
            .create_command_pool_typed(group, flags, max_buffers)
    }

    #[inline]
    fn create_graphics_pipeline<'a>(
        &self,
        desc: &pso::GraphicsPipelineDesc<'a, B>,
    ) -> Result<B::GraphicsPipeline, pso::CreationError> {
        self.device.create_graphics_pipeline(desc)
    }

    #[inline]
    fn create_graphics_pipelines<'a, I>(
        &self,
        descs: I,
    ) -> Vec<Result<B::GraphicsPipeline, pso::CreationError>>
    where
        I: IntoIterator,
        I::Item: Borrow<pso::GraphicsPipelineDesc<'a, B>>,
    {
        self.device.create_graphics_pipelines(descs)
    }

    #[inline]
    fn create_compute_pipeline<'a>(
        &self,
        desc: &pso::ComputePipelineDesc<'a, B>,
    ) -> Result<B::ComputePipeline, pso::CreationError> {
        self.device.create_compute_pipeline(desc)
    }

    #[inline]
    fn create_compute_pipelines<'a, I>(
        &self,
        descs: I,
    ) -> Vec<Result<B::ComputePipeline, pso::CreationError>>
    where
        I: IntoIterator,
        I::Item: Borrow<pso::ComputePipelineDesc<'a, B>>,
    {
        self.device.create_compute_pipelines(descs)
    }

    #[inline]
    fn acquire_mapping_reader<'a, T>(
        &self,
        memory: &'a B::Memory,
        range: Range<u64>,
    ) -> Result<Reader<'a, B, T>, mapping::Error>
    where
        T: Copy,
    {
        self.device.acquire_mapping_reader(memory, range)
    }

    #[inline]
    fn release_mapping_reader<'a, T>(&self, reader: Reader<'a, B, T>) {
        self.device.release_mapping_reader(reader)
    }

    #[inline]
    fn acquire_mapping_writer<'a, T>(
        &self,
        memory: &'a B::Memory,
        range: Range<u64>,
    ) -> Result<Writer<'a, B, T>, mapping::Error>
    where
        T: Copy,
    {
        self.device.acquire_mapping_writer(memory, range)
    }

    #[inline]
    fn release_mapping_writer<'a, T>(&self, writer: Writer<'a, B, T>) {
        self.device.release_mapping_writer(writer)
    }

    #[inline]
    fn reset_fence(&self, fence: &B::Fence) {
        self.device.reset_fence(fence)
    }

    #[inline]
    fn reset_fences<I>(&self, fences: I)
    where
        I: IntoIterator,
        I::Item: Borrow<B::Fence>,
    {
        self.device.reset_fences(fences)
    }

    #[inline]
    fn wait_for_fence(&self, fence: &B::Fence, timeout_ms: u32) -> bool {
        self.device.wait_for_fence(fence, timeout_ms)
    }

    #[inline]
    fn wait_for_fences<I>(&self, fences: I, wait: WaitFor, timeout_ms: u32) -> bool
    where
        I: IntoIterator,
        I::Item: Borrow<B::Fence>,
    {
        self.device.wait_for_fences(fences, wait, timeout_ms)
    }

    #[inline]
    fn get_image_subresource_footprint(
        &self,
        image: &B::Image,
        subresource: image::Subresource,
    ) -> image::SubresourceFootprint {
        self.device
            .get_image_subresource_footprint(image, subresource)
    }
}

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
                device.destroy_buffer(buffer.raw);
                allocator.free(device, buffer.block);
            }
            AnyItem::Image(image) => {
                device.destroy_image(image.raw);
                allocator.free(device, image.block);
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

fn create_relevant_buffer<B>(
    device: &B::Device,
    allocator: &mut SmartAllocator<B>,
    size: u64,
    usage: buffer::Usage,
    alloc_type: Type,
    properties: Properties,
) -> Result<RelevantBuffer<B>, BufferCreationError>
where
    B: Backend,
{
    // Create unbound buffer object.
    let ubuf = device.create_buffer(size, usage)?;

    // Get memory requirements.
    let reqs = device.get_buffer_requirements(&ubuf);

    // Allocate memory block.
    let block = allocator.alloc(device, (alloc_type, properties), reqs)?;

    // Bind memory. Infallible unless unless bugged.
    let raw = device
        .bind_buffer_memory(block.memory(), block.range().start, ubuf)
        .expect("Requirements must be satisfied");

    Ok(RelevantItem { raw, block })
}

fn create_relevant_image<B>(
    device: &B::Device,
    allocator: &mut SmartAllocator<B>,
    kind: image::Kind,
    level: image::Level,
    format: Format,
    tiling: image::Tiling,
    storage_flags: image::StorageFlags,
    usage: image::Usage,
    alloc_type: Type,
    properties: Properties,
) -> Result<RelevantImage<B>, ImageCreationError>
where
    B: Backend,
{
    // Create unbound image object.
    let uimg = device.create_image(kind, level, format, tiling, usage, storage_flags)?;

    // Get memory requirements.
    let reqs = device.get_image_requirements(&uimg);

    // Allocate memory block.
    let block = allocator.alloc(device, (alloc_type, properties), reqs)?;

    // Bind memory. Infallible unless unless bugged.
    let raw = device
        .bind_image_memory(block.memory(), block.range().start, uimg)
        .expect("Requirements must be satisfied");

    Ok(RelevantItem { raw, block })
}
