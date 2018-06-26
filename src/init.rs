use std::marker::PhantomData;

use hal::{
    adapter::{Adapter, PhysicalDevice}, error::DeviceCreationError, queue::QueueFamily, Backend,
    Instance,
};

use backend::BackendEx;
use factory::Factory;
use mem::SmartAllocator;
use renderer::Renderer;

#[cfg(feature = "regex")]
use regex::Regex;

const STAGING_TRESHOLD: usize = 32 * 1024; // 32kb

/// Trait for picking adapter among available.
pub trait AdapterPicker<B: Backend> {
    fn pick_adapter(self, adapters: Vec<Adapter<B>>) -> Option<Adapter<B>>;
}

/// `AdapterPicker` that picks first available adapter.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FirstAdapter;

impl<B> AdapterPicker<B> for FirstAdapter
where
    B: Backend,
{
    fn pick_adapter(self, mut adapters: Vec<Adapter<B>>) -> Option<Adapter<B>> {
        Some(adapters.remove(0))
    }
}

pub struct AdapterFn<B, F>(F, PhantomData<B>);

/// Pick adapter with given closure.
pub fn adapter_picker<B, F>(f: F) -> AdapterFn<B, F>
where
    B: Backend,
    F: FnOnce(Vec<Adapter<B>>) -> Option<Adapter<B>>,
{
    AdapterFn(f, PhantomData)
}

impl<B, F> AdapterPicker<B> for AdapterFn<B, F>
where
    B: Backend,
    F: FnOnce(Vec<Adapter<B>>) -> Option<Adapter<B>>,
{
    fn pick_adapter(self, adapters: Vec<Adapter<B>>) -> Option<Adapter<B>> {
        (self.0)(adapters)
    }
}

/// `AdapterPicker` which searches for adapter with matching name.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdapterByName<T>(T);

impl<B, T> AdapterPicker<B> for AdapterByName<T>
where
    B: Backend,
    T: PartialEq<String>,
{
    fn pick_adapter(self, adapters: Vec<Adapter<B>>) -> Option<Adapter<B>> {
        adapters
            .into_iter()
            .find(|adapter| self.0.eq(&adapter.info.name))
    }
}

/// `AdapterPicker` which searches for adapter with matching name by regex.
#[cfg(feature = "regex")]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdapterByNameRe(#[cfg_attr(feature = "serde", serde(with = "::serde_regex"))] Regex);

#[cfg(feature = "regex")]
impl<B> AdapterPicker<B> for AdapterByNameRe
where
    B: Backend,
{
    fn pick_adapter(self, adapters: Vec<Adapter<B>>) -> Option<Adapter<B>> {
        adapters
            .into_iter()
            .find(|adapter| self.0.is_match(&adapter.info.name))
    }
}

/// Trait for picking queue families among available.
pub trait QueuesPicker<B: Backend> {
    fn pick_queues(self, families: &[B::QueueFamily]) -> Vec<(&B::QueueFamily, usize)>;
}

pub struct QueueFn<B, F>(F, PhantomData<B>);

/// Pick queues with given closure.
pub fn queue_picker<B, F>(f: F) -> QueueFn<B, F>
where
    B: Backend,
    F: FnOnce(&[B::QueueFamily]) -> Vec<(&B::QueueFamily, usize)>,
{
    QueueFn(f, PhantomData)
}

impl<B, F> QueuesPicker<B> for QueueFn<B, F>
where
    B: Backend,
    F: FnOnce(&[B::QueueFamily]) -> Vec<(&B::QueueFamily, usize)>,
{
    fn pick_queues(self, families: &[B::QueueFamily]) -> Vec<(&B::QueueFamily, usize)> {
        (self.0)(families)
    }
}

/// Configuration for memory allocator in `Factory`.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryConfig {
    /// Size of one chunk in arena-based allocator.
    arena_chunk_size: u64,

    /// Blocks per chunk in block-based allocator.
    blocks_per_chunk: usize,

    /// Smallest block size in block-based allocator.
    /// Any lesser allocation will be of this size.
    min_block_size: u64,

    /// Biggest block size in block-based allocator.
    /// Any bigger allocation will allocate directly from gpu.
    max_chunk_size: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            arena_chunk_size: 32,
            blocks_per_chunk: 32,
            min_block_size: 32,
            max_chunk_size: 1024 * 1024 * 32,
        }
    }
}

/// Configuration for initialization.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Config<A, Q> {
    /// `AdapterPicker` implementation.
    /// Will be used to pick from available adapters.
    adapter_picker: A,

    /// `QueuesPicker` implementation.
    /// Will be used to pick from available queue families.
    queues_picker: Q,

    /// Config for allocator.
    memory: MemoryConfig,
}

/// Possible errors during initialization
#[derive(Clone, Debug, Fail)]
pub enum InitError {
    #[fail(display = "Failed to choose from available adapters")]
    AdapterNotChosen,

    #[fail(display = "Failed to create logical device")]
    DeviceCreationError(#[cause] DeviceCreationError),
}

/// Init chosen backend.
/// Creates `Factory` and `Renderer` instances.
pub fn init<B, R, A, Q>(adapter_picker: A, queues_picker: Q, memory: MemoryConfig) -> Result<(Factory<B>, Renderer<B, R>), InitError>
where
    B: BackendEx,
    R: Send + Sync + 'static,
    A: AdapterPicker<B>,
    Q: QueuesPicker<B>,
{
    let instance = B::init();
    let adapter = adapter_picker
        .pick_adapter(instance.enumerate_adapters())
        .ok_or(InitError::AdapterNotChosen)?;
    trace!("Adapter {:#?}", adapter.info);

    trace!("Device features: {:#?}", adapter.physical_device.features());
    trace!("Device limits: {:#?}", adapter.physical_device.limits());

    let (device, queue_groups) = {
        trace!("Queue families: {:#?}", adapter.queue_families);

        let queues = queues_picker
            .pick_queues(&adapter.queue_families)
            .into_iter()
            .map(|(qf, count)| (qf, vec![1.0; count]))
            .collect::<Vec<_>>();
        let queues = queues
            .iter()
            .map(|&(qf, ref priorities)| (qf, priorities.as_slice()))
            .collect::<Vec<_>>();

        let mut gpu = adapter
            .physical_device
            .open(&queues)
            .map_err(InitError::DeviceCreationError)?;
        let queue_groups = queues
            .iter()
            .map(|&(qf, _)| {
                (
                    qf.id(),
                    gpu.queues
                        .take_raw(qf.id())
                        .expect("Family with id was requested"),
                )
            })
            .collect();
        (gpu.device, queue_groups)
    };
    trace!("Logical device created");

    let allocator = SmartAllocator::<B>::new(
        adapter.physical_device.memory_properties(),
        memory.arena_chunk_size,
        memory.blocks_per_chunk,
        memory.min_block_size,
        memory.max_chunk_size,
    );
    trace!("Allocator created: {:#?}", allocator);

    let factory = Factory::new(
        instance,
        adapter.info,
        adapter.physical_device,
        device,
        allocator,
        STAGING_TRESHOLD,
    );
    let renderer = Renderer::<B, R>::new(queue_groups, adapter.queue_families);

    Ok((factory, renderer))
}
