

use failure::Error;

use hal::Instance;
use hal::adapter::PhysicalDevice;
use hal::queue::{General, QueueFamily, QueueType};

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
pub fn init<B, R>() -> Result<(Factory<B>, Renderer<B, R>), Error>
where
    B: BackendEx,
    R: Send + Sync + 'static,
{
    use failure::err_msg;

    let instance = B::init();
    let mut adapter = instance.enumerate_adapters().remove(0);
    trace!("Adapter {:#?}", adapter.info);

    trace!("Device features: {:#?}", adapter.physical_device.features());
    trace!("Device limits: {:#?}", adapter.physical_device.limits());

    let (device, queue_group) = {
        trace!("Queue families: {:#?}", adapter.queue_families);
        let qf = adapter
            .queue_families
            .drain(..)
            .filter(|family| family.queue_type() == QueueType::General)
            .next()
            .ok_or(err_msg("Can't find General queue family"))?;
        let mut gpu = adapter
            .physical_device
            .open(&[(&qf, &[1.0])])?;
        let queue_group = gpu.queues
            .take::<General>(qf.id())
            .expect("This group was requested");
        (gpu.device, queue_group)
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
        adapter.physical_device,
        device,
        allocator,
        STAGING_TRESHOLD,
        queue_group.family(),
    );
    let renderer = Renderer::<B, R>::new(queue_group);

    Ok((factory, renderer))
}
