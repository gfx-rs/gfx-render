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

use failure::{Error, ResultExt};

use hal::{MemoryTypeId, AdapterInfo, Backend, Device, Surface, Limits, Features, PhysicalDevice};
use hal::device::{BindError, ShaderError, FramebufferError, OutOfMemory, WaitFor};
use hal::error::HostExecutionError;
use hal::buffer;
use hal::format::{Format, Swizzle};
use hal::image::{self, Extent, Kind, Layout, Level, Offset, StorageFlags,
                 SubresourceLayers, Subresource, SubresourceRange, SubresourceFootprint, Tiling, ViewKind, SamplerInfo};
use hal::mapping::{self, Reader, Writer};
use hal::memory::{Properties, Requirements};
use hal::pass::{Attachment, SubpassDesc, SubpassDependency};
use hal::pool::{CommandPool, CommandPoolCreateFlags};
use hal::pso::{self, DescriptorRangeDesc, ShaderStageFlags, DescriptorSetLayoutBinding, DescriptorSetWrite, Descriptor, DescriptorSetCopy, GraphicsPipelineDesc, ComputePipelineDesc};
use hal::range::RangeArg;
use hal::queue::{QueueGroup, QueueFamilyId};
use hal::query::QueryType;
use hal::window::{Backbuffer, SwapchainConfig, SurfaceCapabilities, Extent2D};

use mem::{Block, Factory as FactoryTrait, SmartAllocator, SmartBlock, Type, MemoryAllocator};

use winit::Window;

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
pub type Buffer<B> = Item<<B as Backend>::Buffer, SmartBlock<<B as Backend>::Memory>>;

/// Image type `Factory` creates
pub type Image<B> = Item<<B as Backend>::Image, SmartBlock<<B as Backend>::Memory>>;

/// `Factory` is a central type that wraps GPU device and responsible for:
/// 1. Creating new `Buffer`s and `Image`s.
/// 2. Destroying `Buffer`s and `Image`s with additional safety.
/// 3. Uploading data to `Buffer`s and `Image`s.
/// 4. Creating `Surface`s and fetching their capabilities.
/// 5. Fetching `Features` and `Limits` of the GPU.
///
#[repr(C)]
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
            debug!("Dispose of uploader.");
            read(&mut *self.upload).dispose(&self.device);

            debug!("Advance to the end of times");
            self.advance();

            debug!("Dispose of allocator.");
            read(&mut *self.allocator).dispose(&self.device).expect("Allocator must be cleared");
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
        properties: Properties,
        usage: buffer::Usage,
    ) -> Result<Buffer<B>, Error> {
        let buffer: RelevantBuffer<B> = self.allocator
            .create_buffer(
                self.device.borrow(),
                (Type::General, properties),
                size,
                usage,
            )
            .with_context(|_| "Failed to create buffer")?;
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
        usage: image::Usage,
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
            .with_context(|_| "Failed to create image")?;
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
        buffer: &mut Buffer<B>,
        family: QueueFamilyId,
        access: buffer::Access,
        offset: u64,
        data: &[u8],
    ) -> Result<(), Error> {
        let ref device = self.device;
        let ref mut allocator = self.allocator;
        if let Some(staging) =
            self.upload
                .upload_buffer(device, allocator, &mut *buffer.inner, family, access, offset, data)?
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
        family: QueueFamilyId,
        layout: Layout,
        access: image::Access,
        layers: SubresourceLayers,
        offset: Offset,
        extent: Extent,
        data_width: u32,
        data_height: u32,
        data: &[u8],
    ) -> Result<(), Error> {
        let ref device = self.device;
        let ref mut allocator = self.allocator;
        let staging = self.upload.upload_image(
            device,
            allocator,
            &mut *image.inner,
            family,
            access,
            layout,
            layers,
            offset,
            extent,
            data_width,
            data_height,
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
    pub fn capabilities_and_formats(
        &self,
        surface: &B::Surface,
    ) -> (SurfaceCapabilities, Option<Vec<Format>>) {
        surface.capabilities_and_formats(&self.physical_device)
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
    pub(crate) fn uploads(&mut self) -> impl Iterator<Item = (&mut B::CommandBuffer, QueueFamilyId)> {
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
        size: u64
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
        create_flags: CommandPoolCreateFlags
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
        dependencies: ID
    ) -> B::RenderPass
    where
        IA: IntoIterator,
        IA::Item: Borrow<Attachment>,
        IS: IntoIterator,
        IS::Item: Borrow<SubpassDesc<'a>>,
        ID: IntoIterator,
        ID::Item: Borrow<SubpassDependency>
    {
        self.device.create_render_pass(
            attachments,
            subpasses,
            dependencies,
        )
    }

    #[inline]
    fn destroy_render_pass(&self, rp: B::RenderPass) {
        self.device.destroy_render_pass(rp)
    }

    #[inline]
    fn create_pipeline_layout<IS, IR>(
        &self, 
        set_layouts: IS, 
        push_constant: IR
    ) -> B::PipelineLayout
    where
        IS: IntoIterator,
        IS::Item: Borrow<B::DescriptorSetLayout>,
        IR: IntoIterator,
        IR::Item: Borrow<(ShaderStageFlags, Range<u32>)>
    {
        self.device.create_pipeline_layout(
            set_layouts,
            push_constant,
        )
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
        extent: Extent
    ) -> Result<B::Framebuffer, FramebufferError>
    where
        I: IntoIterator,
        I::Item: Borrow<B::ImageView>
    {
        self.device.create_framebuffer(pass, attachments, extent)
    }

    #[inline]
    fn destroy_framebuffer(&self, buf: B::Framebuffer) {
        self.device.destroy_framebuffer(buf)
    }

    #[inline]
    fn create_shader_module(
        &self, 
        spirv_data: &[u8]
    ) -> Result<B::ShaderModule, ShaderError> {
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
        usage: buffer::Usage
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
        buf: B::UnboundBuffer
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
        range: R
    ) -> Result<B::BufferView, buffer::ViewError> {
        self.device.create_buffer_view(buf, fmt, range)
    }

    #[inline]
    fn destroy_buffer_view(&self, view: B::BufferView) {
        self.device.destroy_buffer_view(view)
    }

    #[inline]
    fn create_image(
        &self, 
        kind: Kind, 
        mip_levels: Level, 
        format: Format, 
        tiling: Tiling, 
        usage: image::Usage, 
        storage_flags: StorageFlags
    ) -> Result<B::UnboundImage, image::CreationError> {
        self.device.create_image(kind, mip_levels, format, tiling, usage, storage_flags)
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
        image: B::UnboundImage
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
        view_kind: ViewKind, 
        format: Format, 
        swizzle: Swizzle, 
        range: SubresourceRange
    ) -> Result<B::ImageView, image::ViewError> {
        self.device.create_image_view(image, view_kind, format, swizzle, range)
    }

    #[inline]
    fn destroy_image_view(&self, view: B::ImageView) {
        self.device.destroy_image_view(view)
    }

    #[inline]
    fn create_sampler(&self, info: SamplerInfo) -> B::Sampler {
        self.device.create_sampler(info)
    }

    #[inline]
    fn destroy_sampler(&self, sampler: B::Sampler) {
        self.device.destroy_sampler(sampler)
    }

    #[inline]
    fn create_descriptor_pool<I>(
        &self, 
        max_sets: usize, 
        descriptor_ranges: I
    ) -> B::DescriptorPool
    where
        I: IntoIterator,
        I::Item: Borrow<DescriptorRangeDesc>,
    {
        self.device.create_descriptor_pool(max_sets, descriptor_ranges)
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
        I::Item: Borrow<DescriptorSetLayoutBinding>,
        J: IntoIterator,
        J::Item: Borrow<B::Sampler>,
    {
        self.device.create_descriptor_set_layout(bindings, immutable_samplers)
    }

    #[inline]
    fn destroy_descriptor_set_layout(&self, layout: B::DescriptorSetLayout) {
        self.device.destroy_descriptor_set_layout(layout)
    }

    #[inline]
    fn write_descriptor_sets<'a, I, J>(&self, write_iter: I)
    where
        I: IntoIterator<Item = DescriptorSetWrite<'a, B, J>>,
        J: IntoIterator,
        J::Item: Borrow<Descriptor<'a, B>>,
    {
        self.device.write_descriptor_sets(write_iter)
    }

    #[inline]
    fn copy_descriptor_sets<'a, I>(&self, copy_iter: I)
    where
        I: IntoIterator,
        I::Item: Borrow<DescriptorSetCopy<'a, B>>,
    {
        self.device.copy_descriptor_sets(copy_iter)
    }

    #[inline]
    fn map_memory<R>(
        &self, 
        memory: &B::Memory, 
        range: R
    ) -> Result<*mut u8, mapping::Error>
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
        config: SwapchainConfig,
        old_swapchain: Option<B::Swapchain>,
        extent: &Extent2D
    ) -> (B::Swapchain, Backbuffer<B>) {
        self.device.create_swapchain(surface, config, old_swapchain, extent)
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
        max_buffers: usize
    ) -> CommandPool<B, C> {
        self.device.create_command_pool_typed(group, flags, max_buffers)
    }

    #[inline]
    fn create_graphics_pipeline<'a>(
        &self,
        desc: &GraphicsPipelineDesc<'a, B>
    ) -> Result<B::GraphicsPipeline, pso::CreationError> {
        self.device.create_graphics_pipeline(desc)
    }

    #[inline]
    fn create_graphics_pipelines<'a, I>(
        &self,
        descs: I
    ) -> Vec<Result<B::GraphicsPipeline, pso::CreationError>>
    where
        I: IntoIterator,
        I::Item: Borrow<GraphicsPipelineDesc<'a, B>>,
    {
        self.device.create_graphics_pipelines(descs)
    }

    #[inline]
    fn create_compute_pipeline<'a>(
        &self, 
        desc: &ComputePipelineDesc<'a, B>
    ) -> Result<B::ComputePipeline, pso::CreationError> {
        self.device.create_compute_pipeline(desc)
    }

    #[inline]
    fn create_compute_pipelines<'a, I>(
        &self, 
        descs: I
    ) -> Vec<Result<B::ComputePipeline, pso::CreationError>>
    where
        I: IntoIterator,
        I::Item: Borrow<ComputePipelineDesc<'a, B>>,
    {
        self.device.create_compute_pipelines(descs)
    }

    #[inline]
    fn acquire_mapping_reader<'a, T>(
        &self, 
        memory: &'a B::Memory, 
        range: Range<u64>
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
        range: Range<u64>
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
    fn wait_for_fences<I>(
        &self, 
        fences: I, 
        wait: WaitFor, 
        timeout_ms: u32
    ) -> bool
    where
        I: IntoIterator,
        I::Item: Borrow<B::Fence>,
    {
        self.device.wait_for_fences(fences, wait, timeout_ms)
    }

    #[inline]
    fn get_image_subresource_footprint(
        &self, image: &B::Image, subresource: Subresource
    ) -> SubresourceFootprint {
        self.device.get_image_subresource_footprint(image, subresource)
    }
}

pub(crate) type RelevantBuffer<B> = RelevantItem<<B as Backend>::Buffer, SmartBlock<<B as Backend>::Memory>>;
pub(crate) type RelevantImage<B> = RelevantItem<<B as Backend>::Image, SmartBlock<<B as Backend>::Memory>>;

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
