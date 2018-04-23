use std::borrow::{Borrow, BorrowMut};
use std::collections::VecDeque;
use std::slice::from_raw_parts_mut;

use failure::{Error, Fail, ResultExt};

use hal::{Backend, Device};
use hal::buffer::{Access as BufferAccess, Usage as BufferUsage};
use hal::command::{BufferCopy, BufferImageCopy, CommandBufferFlags, RawCommandBuffer, RawLevel};

use hal::image::{Access as ImageAccess, Extent, Layout, Offset, SubresourceLayers,
                 SubresourceRange};
use hal::mapping::Error as MappingError;
use hal::memory::{Barrier, Dependencies, Properties};
use hal::pool::{CommandPoolCreateFlags, RawCommandPool};
use hal::pso::PipelineStage;
use hal::queue::QueueFamilyId;

use mem::{Block, Factory, Item, SmartAllocator, SmartBlock, Type};

type SmartBuffer<B: Backend> = Item<B::Buffer, SmartBlock<B::Memory>>;
type SmartImage<B: Backend> = Item<B::Image, SmartBlock<B::Memory>>;

#[derive(Debug)]
pub struct Upload<B: Backend> {
    staging_threshold: usize,
    family: QueueFamilyId,
    pool: Option<B::CommandPool>,
    cbuf: Option<B::CommandBuffer>,
    free: Vec<B::CommandBuffer>,
    used: VecDeque<(B::CommandBuffer, u64)>,
}

impl<B> Upload<B>
where
    B: Backend,
{
    pub fn dispose(mut self, device: &B::Device) {
        self.clear(u64::max_value());
        self.free.clear();
        if let Some(mut pool) = self.pool {
            pool.reset();
            device.destroy_command_pool(pool);
        }
        assert!(self.used.is_empty());
    }

    pub fn new(staging_threshold: usize, family: QueueFamilyId) -> Self {
        Upload {
            staging_threshold,
            family,
            pool: None,
            cbuf: None,
            free: Vec::new(),
            used: VecDeque::new(),
        }
    }

    pub fn upload_buffer(
        &mut self,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
        buffer: &mut SmartBuffer<B>,
        access: BufferAccess,
        offset: u64,
        data: &[u8],
    ) -> Result<Option<SmartBuffer<B>>, Error> {
        if buffer.size() < offset + data.len() as u64 {
            return Err(MappingError::OutOfBounds.context("Buffer upload failed").into())
        }
        let props = allocator.properties(buffer.block());
        if props.contains(Properties::CPU_VISIBLE) {
            unsafe {
                // Safe due to block is checked to have `CPU_VISIBLE` property.
                update_cpu_visible_block::<B>(
                    device,
                    props.contains(Properties::COHERENT),
                    buffer.block(),
                    offset,
                    data,
                );
            }
            Ok(None)
        } else {
            self.upload_device_local_buffer(device, allocator, buffer, access, offset, data)
        }
    }

    pub fn upload_image(
        &mut self,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
        image: &mut SmartImage<B>,
        access: ImageAccess,
        layout: Layout,
        layers: SubresourceLayers,
        offset: Offset,
        extent: Extent,
        data_width: u32,
        data_height: u32,
        data: &[u8],
    ) -> Result<SmartBuffer<B>, Error> {
        // Check requirements.
        // TODO: Return `Error` instead of panicking.
        assert!(data_width >= extent.width);
        assert!(data_height >= extent.height);
        assert_eq!(layers.aspects.bits().count_ones(), 1);
        assert!(data_width as usize * data_height as usize * extent.depth as usize <= data.len());

        let staging = allocator
            .create_buffer(
                device,
                (Type::ShortLived, Properties::CPU_VISIBLE),
                data.len() as u64,
                BufferUsage::TRANSFER_SRC,
            )
            .with_context(|_| "Failed to create staging buffer")?;
        let props = allocator.properties(staging.block());
        unsafe {
            // Safe due to block is allocated with `CPU_VISIBLE` property.
            update_cpu_visible_block::<B>(
                device,
                props.contains(Properties::COHERENT),
                staging.block(),
                0,
                data,
            );
        }

        let uploading_layout = if layout == Layout::General {
            Layout::General
        } else {
            Layout::TransferDstOptimal
        };

        let cbuf = self.get_command_buffer(device);
        cbuf.copy_buffer_to_image(
            staging.borrow(),
            image.borrow_mut(),
            uploading_layout,
            Some(BufferImageCopy {
                buffer_offset: 0,
                buffer_width: data_width,
                buffer_height: data_height,
                image_layers: layers.clone(),
                image_offset: offset,
                image_extent: extent,
            }),
        );

        cbuf.pipeline_barrier(
            PipelineStage::TRANSFER..PipelineStage::TOP_OF_PIPE,
            Dependencies::empty(),
            Some(Barrier::Image {
                states: (ImageAccess::TRANSFER_WRITE, uploading_layout)..(access, layout),
                target: image.borrow_mut(),
                range: SubresourceRange {
                    aspects: layers.aspects,
                    levels: layers.level..layers.level,
                    layers: layers.layers,
                },
            }),
        );
        Ok(staging)
    }

    pub fn uploads(&mut self, frame: u64) -> Option<(&mut B::CommandBuffer, QueueFamilyId)> {
        if let Some(mut cbuf) = self.cbuf.take() {
            cbuf.finish();
            self.used.push_back((cbuf, frame));
            Some((&mut self.used.back_mut().unwrap().0, self.family))
        } else {
            None
        }
    }

    pub fn clear(&mut self, ongoing: u64) {
        while let Some((mut cbuf, frame)) = self.used.pop_front() {
            if frame >= ongoing {
                self.used.push_front((cbuf, ongoing));
                break;
            }
            cbuf.reset(true);
            self.free.push(cbuf);
        }
    }

    fn get_command_buffer<'a>(&'a mut self, device: &B::Device) -> &'a mut B::CommandBuffer {
        let Upload {
            family,
            ref mut pool,
            ref mut free,
            ref mut cbuf,
            ..
        } = *self;
        cbuf.get_or_insert_with(|| {
            let mut cbuf = free.pop().unwrap_or_else(|| {
                let pool = pool.get_or_insert_with(|| {
                    device.create_command_pool(family, CommandPoolCreateFlags::RESET_INDIVIDUAL)
                });
                pool.allocate(1, RawLevel::Primary).remove(0)
            });
            cbuf.begin(CommandBufferFlags::ONE_TIME_SUBMIT, Default::default());
            cbuf
        })
    }

    fn upload_device_local_buffer(
        &mut self,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
        buffer: &mut SmartBuffer<B>,
        access: BufferAccess,
        offset: u64,
        data: &[u8],
    ) -> Result<Option<SmartBuffer<B>>, Error> {
        if data.len() <= self.staging_threshold {
            let cbuf = self.get_command_buffer(device);
            cbuf.update_buffer(buffer.borrow_mut(), offset, data);

            cbuf.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::TOP_OF_PIPE,
                Dependencies::empty(),
                Some(Barrier::Buffer {
                    states: BufferAccess::TRANSFER_WRITE..access,
                    target: buffer.borrow_mut(),
                }),
            );
            Ok(None)
        } else {
            let staging = allocator
                .create_buffer(
                    device,
                    (Type::ShortLived, Properties::CPU_VISIBLE),
                    data.len() as u64,
                    BufferUsage::TRANSFER_SRC,
                )
                .with_context(|_| "Failed to create staging buffer")?;
            let props = allocator.properties(staging.block());
            unsafe {
                // Safe due to block is allocated with `CPU_VISIBLE` property.
                update_cpu_visible_block::<B>(
                    device,
                    props.contains(Properties::COHERENT),
                    staging.block(),
                    0,
                    data,
                );
            }
            let cbuf = self.get_command_buffer(device);
            cbuf.copy_buffer(
                staging.borrow(),
                (&*buffer).borrow(),
                Some(BufferCopy {
                    src: 0,
                    dst: offset,
                    size: data.len() as u64,
                }),
            );
            cbuf.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::TOP_OF_PIPE,
                Dependencies::empty(),
                Some(Barrier::Buffer {
                    states: BufferAccess::TRANSFER_WRITE..access,
                    target: buffer.borrow_mut(),
                }),
            );
            Ok(Some(staging))
        }
    }
}

/// Update cpu-visible block.
///
/// # Safety
///
/// Caller must be sure that memory of the block has `CPU_VISIBLE` property.
/// `coherent` argument must be set to `true` only if memory of the block has `COHERENT` property.
///
pub unsafe fn update_cpu_visible_block<B: Backend>(
    device: &B::Device,
    coherent: bool,
    block: &SmartBlock<B::Memory>,
    offset: u64,
    data: &[u8],
) {
    let start = block.range().start + offset;
    let end = start + data.len() as u64;
    let range = start..end;
    debug_assert!(
        end <= block.range().end,
        "Checked in `Upload::upload` method"
    );
    let ptr = device
        .map_memory(block.memory(), range.clone())
        .expect("Expect to be mapped");
    if !coherent {
        device.invalidate_mapped_memory_ranges(Some((block.memory(), range.clone())));
    }
    let slice = from_raw_parts_mut(ptr, data.len());
    slice.copy_from_slice(data);
    if !coherent {
        device.flush_mapped_memory_ranges(Some((block.memory(), range)));
    }
}
