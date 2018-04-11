//! This module provide extension trait `BackendEx` that is implemented for some gfx backends:
//! vulkan, dx12 and metal if corresponding feature is enabled.
//! Also empty backend implements `BackendEx` if no backend-features enabled.
//!

use hal::{Backend, Instance};
use winit::Window;

#[cfg(feature = "gfx-backend-vulkan")]
use vulkan;

#[cfg(feature = "gfx-backend-metal")]
use metal;

#[cfg(feature = "gfx-backend-dx12")]
use dx12;

#[cfg(not(any(feature = "gfx-backend-vulkan", feature = "gfx-backend-metal",
              feature = "gfx-backend-dx12")))]
use empty;

/// Extend backend trait with initialization method and surface creation method.
pub trait BackendEx: Backend {
    type Instance: Instance<Backend = Self> + Send + Sync;
    fn init() -> Self::Instance;
    fn create_surface(instance: &Self::Instance, window: &Window) -> Self::Surface;
}

#[cfg(feature = "gfx-backend-vulkan")]
impl BackendEx for vulkan::Backend {
    type Instance = vulkan::Instance;
    fn init() -> Self::Instance {
        vulkan::Instance::create("amethyst", 1)
    }
    fn create_surface(instance: &Self::Instance, window: &Window) -> Self::Surface {
        instance.create_surface(window)
    }
}

#[cfg(feature = "gfx-backend-metal")]
impl BackendEx for metal::Backend {
    type Instance = metal::Instance;
    fn init() -> Self::Instance {
        metal::Instance::create("amethyst", 1)
    }
    fn create_surface(instance: &Self::Instance, window: &Window) -> Self::Surface {
        instance.create_surface(window)
    }
}

#[cfg(feature = "gfx-backend-dx12")]
impl BackendEx for dx12::Backend {
    type Instance = dx12::Instance;
    fn init() -> Self::Instance {
        dx12::Instance::create("amethyst", 1)
    }
    fn create_surface(instance: &Self::Instance, window: &Window) -> Self::Surface {
        instance.create_surface(window)
    }
}

#[cfg(not(any(feature = "gfx-backend-vulkan", feature = "gfx-backend-metal",
              feature = "gfx-backend-dx12")))]
impl BackendEx for empty::Backend {
    type Instance = empty::Instance;
    fn init() -> Self::Instance {
        empty::Instance
    }
    fn create_surface(_: &Self::Instance, _: &Window) -> Self::Surface {
        empty::Surface
    }
}
