//! Module provides wrapper for types that cannot be dropped silently.
//! Usually such types are required to be returned to their creator.
//! `Escape` wrapper help the user to do so by sending underlying value to the `Terminal` when it is dropped.
//! Users are encouraged to dispose of the values manually while `Escape` be just a safety net.

use std::mem::{forget, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::ptr::read;
use crossbeam_channel::{unbounded, Receiver, Sender, TryIter, TryRecvError};

/// Wraps value of any type and send it to the `Terminal` from which the wrapper was created.
/// In case `Terminal` is already dropped then value will be cast into oblivion via `std::mem::forget`.
#[derive(Debug, Clone)]
pub struct Escape<T> {
    value: ManuallyDrop<T>,
    sender: Sender<T>,
}

impl<T> Escape<T> {
    /// Unwrap the value.
    pub fn into_inner(escape: Self) -> T {
        Self::deconstruct(escape).0
    }

    fn deconstruct(mut escape: Self) -> (T, Sender<T>) {
        unsafe {
            let value = read(&mut *escape.value);
            let sender = read(&mut escape.sender);
            forget(escape);
            (value, sender)
        }
    }
}

impl<T> Deref for Escape<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &*self.value
    }
}

impl<T> DerefMut for Escape<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.value
    }
}

impl<T> Drop for Escape<T> {
    fn drop(&mut self) {
        let value = unsafe { read(&mut *self.value) };
        self.sender
            .send(value)
            .unwrap_or_else(|value| forget(value));
    }
}

/// This types allows the user to create `Escape` wrappers.
/// Receives values from dropped `Escape` instances that was created by this `Terminal`.
#[derive(Debug)]
pub struct Terminal<T> {
    receiver: Receiver<T>,
    sender: ManuallyDrop<Sender<T>>,
}

impl<T> Default for Terminal<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Terminal<T> {
    /// Create new `Terminal`.
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Terminal { sender: ManuallyDrop::new(sender), receiver }
    }

    /// Wrap the value. It will be yielded by iterator returned by `Terminal::drain` if `Escape` will be dropped.
    pub fn escape(&self, value: T) -> Escape<T> {
        Escape {
            value: ManuallyDrop::new(value),
            sender: Sender::clone(&self.sender),
        }
    }

    /// Get iterator over values from dropped `Escape` instances that was created by this `Terminal`.
    pub fn drain(&mut self) -> TryIter<T> {
        self.receiver.try_iter()
    }
}

impl<T> Drop for Terminal<T> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.sender);
            match self.receiver.try_recv() {
                Err(TryRecvError::Disconnected) => {}
                _ => {
                    panic!("Terminal must be dropped after all `Escape`s");
                }
            }
        }
    }
}
