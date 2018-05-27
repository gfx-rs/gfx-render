

use failure::Error;

use hal::Instance;
use hal::adapter::PhysicalDevice;
use hal::queue::QueueFamily;

use mem::SmartAllocator;
use backend::BackendEx;
use factory::Factory;
use renderer::Renderer;

const STAGING_TRESHOLD: usize = 32 * 1024; // 32kb

/// Init chosen backend and create `Factory` and `Renderer` instances.
///
/// # TODO
///
/// Add config.
///
pub fn init<B, R, F>(queues: F) -> Result<(Factory<B>, Renderer<B, R>), Error>
where
    B: BackendEx,
    R: Send + Sync + 'static,
    F: FnOnce(&[B::QueueFamily]) -> Vec<(&B::QueueFamily, usize)>
{
    let instance = B::init();
    let adapter = instance.enumerate_adapters().remove(0);
    trace!("Adapter {:#?}", adapter.info);

    trace!("Device features: {:#?}", adapter.physical_device.features());
    trace!("Device limits: {:#?}", adapter.physical_device.limits());

    let (device, queue_groups) = {
        trace!("Queue families: {:#?}", adapter.queue_families);

        let queues = queues(&adapter.queue_families).into_iter().map(|(qf, count)| (qf, vec![1.0; count])).collect::<Vec<_>>();
        let queues = queues.iter().map(|&(qf, ref priorities)| (qf, priorities.as_slice())).collect::<Vec<_>>();

        let mut gpu = adapter
            .physical_device
            .open(&queues)?;
        let queue_groups = queues.iter().map(|&(qf, _)| (qf.id(), gpu.queues.take_raw(qf.id()).unwrap())).collect();
        (gpu.device, queue_groups)
    };
    trace!("Logical device created");

    let allocator = SmartAllocator::<B>::new(
        adapter.physical_device.memory_properties(),
        32,
        32,
        32,
        1024 * 1024 * 64,
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
