use std::borrow::{Borrow, BorrowMut};
use std::collections::{VecDeque, HashMap};
use std::slice::from_raw_parts_mut;
use std::fmt;

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

type SmartBuffer<B> = Item<<B as Backend>::Buffer, SmartBlock<<B as Backend>::Memory>>;
type SmartImage<B> = Item<<B as Backend>::Image, SmartBlock<<B as Backend>::Memory>>;


#[derive(Debug)]
struct FamilyDebug {
    cbuf: bool,
    free: usize,
    used: usize,
}

struct Family<B: Backend> {
    pool: B::CommandPool,
    cbuf: Option<B::CommandBuffer>,
    free: Vec<B::CommandBuffer>,
    used: VecDeque<(B::CommandBuffer, u64)>,
}

impl<B> fmt::Debug for Family<B>
where
    B: Backend,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&FamilyDebug {
            cbuf: self.cbuf.is_some(),
            free: self.free.len(),
            used: self.used.len(),
        }, fmt)
    }
}

#[derive(Debug)]
pub struct Upload<B: Backend> {
    staging_threshold: usize,
    families: HashMap<QueueFamilyId, Family<B>>,
}

impl<B> Upload<B>
where
    B: Backend,
{
    pub fn dispose(mut self, device: &B::Device) {
        self.clear(u64::max_value());

        for (_, mut family) in self.families {
            family.free.clear();
            assert!(family.used.is_empty());
            unsafe {
                family.pool.free(family.cbuf.into_iter().chain(family.free).collect());
            }
            device.destroy_command_pool(family.pool);
        }
    }

    pub fn new(staging_threshold: usize) -> Self {
        Upload {
            staging_threshold,
            families: HashMap::new(),
        }
    }

    pub fn upload_buffer(
        &mut self,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
        buffer: &mut SmartBuffer<B>,
        fid: QueueFamilyId,
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
            self.upload_device_local_buffer(device, allocator, buffer, fid, access, offset, data)
        }
    }

    pub fn upload_image(
        &mut self,
        device: &B::Device,
        allocator: &mut SmartAllocator<B>,
        image: &mut SmartImage<B>,
        fid: QueueFamilyId,
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

        let cbuf = self.get_command_buffer(device, fid);
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

    #[inline]
    pub fn uploads(&mut self, frame: u64) -> impl Iterator<Item = (&mut B::CommandBuffer, QueueFamilyId)> {
        self.families.iter_mut().filter_map(move |(&fid, family)| {
            if let Some(mut cbuf) = family.cbuf.take() {
                cbuf.finish();
                family.used.push_back((cbuf, frame));
                Some((&mut family.used.back_mut().unwrap().0, fid))
            } else {
                None
            }
        })
    }

    pub fn clear(&mut self, ongoing: u64) {
        self.families.iter_mut().for_each(|(_, family)| {
            while let Some((mut cbuf, frame)) = family.used.pop_front() {
                if frame >= ongoing {
                    family.used.push_front((cbuf, ongoing));
                    break;
                }
                cbuf.reset(true);
                family.free.push(cbuf);
            }
        });
    }

    fn get_command_buffer<'a>(&'a mut self, device: &B::Device, fid: QueueFamilyId) -> &'a mut B::CommandBuffer {
        let Family {
            ref mut pool,
            ref mut cbuf,
            ref mut free,
            ..
        } = *self.families.entry(fid).or_insert(Family {
            pool: device.create_command_pool(fid, CommandPoolCreateFlags::RESET_INDIVIDUAL),
            cbuf: None,
            free: Vec::new(),
            used: VecDeque::new(),
        });
        cbuf.get_or_insert_with(|| {
            let mut cbuf = free.pop().unwrap_or_else(|| {
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
        fid: QueueFamilyId,
        access: BufferAccess,
        offset: u64,
        data: &[u8],
    ) -> Result<Option<SmartBuffer<B>>, Error> {
        if data.len() <= self.staging_threshold {
            let cbuf = self.get_command_buffer(device, fid);
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
            let cbuf = self.get_command_buffer(device, fid);
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
