//! # Module: memory_pool
//!
//! ## Responsibility
//! Fixed-size slab memory pool for agent message buffers with use-after-free
//! protection via generational handles.
//!
//! Provides:
//! - [`SlabHandle`]: a generational handle into a [`Slab`]; invalid after
//!   [`Slab::free`] is called.
//! - [`Slab<T>`]: pre-allocated, generation-tracked slab allocator.
//! - [`MessageBuffer`]: a fixed-capacity byte buffer for agent messages.
//! - [`MessagePool`]: a [`Slab<MessageBuffer>`] with domain-specific helpers.
//!
//! ## Guarantees
//! - Use-after-free is detected: each slot carries a monotonically increasing
//!   generation counter; a handle is invalid if its generation differs from the
//!   slot's current generation.
//! - Pool exhaustion: [`Slab::alloc`] returns `None` when no free slots remain.
//! - No `.unwrap()` or `.expect()` in production paths.

// ── SlabHandle ────────────────────────────────────────────────────────────────

/// A handle into a [`Slab`].
///
/// Handles are invalidated when the slot they point to is freed, preventing
/// use-after-free bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlabHandle {
    /// Index of the slot within the slab.
    pub index: usize,
    /// Generation counter at the time of allocation; must match the slot's
    /// current generation for the handle to be valid.
    pub generation: u64,
}

// ── SlabSlot ──────────────────────────────────────────────────────────────────

/// Internal slot held by a [`Slab`].
struct SlabSlot<T> {
    /// Current generation of this slot; incremented on each free.
    generation: u64,
    /// The stored value when the slot is occupied.
    value: Option<T>,
}

impl<T> SlabSlot<T> {
    fn new() -> Self {
        SlabSlot {
            generation: 0,
            value: None,
        }
    }
}

// ── Slab ──────────────────────────────────────────────────────────────────────

/// Pre-allocated slab allocator with generational use-after-free protection.
///
/// Slots are pre-allocated at construction time; allocation and deallocation
/// are O(1) amortised (free-list pop/push).
pub struct Slab<T> {
    slots: Vec<SlabSlot<T>>,
    free_list: Vec<usize>,
    allocated: usize,
}

impl<T> Slab<T> {
    /// Create a new [`Slab`] with `capacity` pre-allocated slots.
    pub fn new(capacity: usize) -> Self {
        let slots: Vec<SlabSlot<T>> = (0..capacity).map(|_| SlabSlot::new()).collect();
        let free_list: Vec<usize> = (0..capacity).rev().collect();
        Slab {
            slots,
            free_list,
            allocated: 0,
        }
    }

    /// Allocate a slot and return a [`SlabHandle`].
    ///
    /// Returns `None` if the pool is exhausted.
    pub fn alloc(&mut self, value: T) -> Option<SlabHandle> {
        let index = self.free_list.pop()?;
        let slot = &mut self.slots[index];
        slot.value = Some(value);
        self.allocated += 1;
        Some(SlabHandle {
            index,
            generation: slot.generation,
        })
    }

    /// Free the slot pointed to by `handle`.
    ///
    /// Increments the slot generation so existing handles become invalid.
    /// Does nothing if the handle is already stale.
    pub fn free(&mut self, handle: SlabHandle) {
        if let Some(slot) = self.slots.get_mut(handle.index) {
            if slot.generation == handle.generation {
                slot.value = None;
                slot.generation = slot.generation.wrapping_add(1);
                self.free_list.push(handle.index);
                self.allocated = self.allocated.saturating_sub(1);
            }
        }
    }

    /// Return a shared reference to the value at `handle`, or `None` if the
    /// handle is stale or the index is out of range.
    pub fn get(&self, handle: SlabHandle) -> Option<&T> {
        let slot = self.slots.get(handle.index)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_ref()
    }

    /// Return a mutable reference to the value at `handle`, or `None` if
    /// stale.
    pub fn get_mut(&mut self, handle: SlabHandle) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_mut()
    }

    /// Total number of slots in the slab.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of currently allocated (occupied) slots.
    pub fn allocated(&self) -> usize {
        self.allocated
    }

    /// Number of available (free) slots.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }
}

// ── MessageBuffer ─────────────────────────────────────────────────────────────

/// A fixed-capacity byte buffer used as an agent message buffer.
#[derive(Debug)]
pub struct MessageBuffer {
    data: Vec<u8>,
    /// Number of valid bytes currently stored.
    pub len: usize,
    /// Maximum bytes this buffer can hold.
    pub capacity: usize,
}

impl MessageBuffer {
    /// Create a new [`MessageBuffer`] with the given capacity.
    pub fn new(capacity: usize) -> Self {
        MessageBuffer {
            data: vec![0u8; capacity],
            len: 0,
            capacity,
        }
    }

    /// Write `bytes` into the buffer.
    ///
    /// Writes are capped at `capacity - len` bytes.  Returns the number of
    /// bytes actually written.
    pub fn write(&mut self, bytes: &[u8]) -> usize {
        let available = self.capacity.saturating_sub(self.len);
        let to_write = bytes.len().min(available);
        self.data[self.len..self.len + to_write].copy_from_slice(&bytes[..to_write]);
        self.len += to_write;
        to_write
    }

    /// Return a slice of the valid bytes currently stored in the buffer.
    pub fn read(&self) -> &[u8] {
        &self.data[..self.len]
    }

    /// Reset the buffer to empty (does not zero memory).
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

// ── MessagePool ───────────────────────────────────────────────────────────────

/// A pool of [`MessageBuffer`]s backed by a [`Slab`].
///
/// All buffers share the same fixed per-message capacity configured at
/// construction time.
pub struct MessagePool {
    slab: Slab<MessageBuffer>,
    /// Capacity of each [`MessageBuffer`] in this pool, in bytes.
    message_capacity: usize,
}

impl MessagePool {
    /// Create a new [`MessagePool`] with `pool_size` buffers each of
    /// `message_capacity` bytes.
    pub fn new(pool_size: usize, message_capacity: usize) -> Self {
        MessagePool {
            slab: Slab::new(pool_size),
            message_capacity,
        }
    }

    /// Acquire a [`MessageBuffer`] from the pool.
    ///
    /// Returns `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Option<SlabHandle> {
        self.slab.alloc(MessageBuffer::new(self.message_capacity))
    }

    /// Return a buffer to the pool.
    pub fn release(&mut self, handle: SlabHandle) {
        // Clear the buffer before returning it.
        if let Some(buf) = self.slab.get_mut(handle) {
            buf.clear();
        }
        self.slab.free(handle);
    }

    /// Return a mutable reference to the [`MessageBuffer`] at `handle`.
    ///
    /// Returns `None` if the handle is stale.
    pub fn get_buffer(&mut self, handle: SlabHandle) -> Option<&mut MessageBuffer> {
        self.slab.get_mut(handle)
    }

    /// Number of buffers currently in use.
    pub fn allocated(&self) -> usize {
        self.slab.allocated()
    }

    /// Number of available buffers.
    pub fn free_count(&self) -> usize {
        self.slab.free_count()
    }

    /// Total pool capacity.
    pub fn capacity(&self) -> usize {
        self.slab.capacity()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    // ── Slab tests ────────────────────────────────────────────────────────────

    #[test]
    fn alloc_and_free_roundtrip() {
        let mut slab: Slab<u32> = Slab::new(4);
        assert_eq!(slab.capacity(), 4);
        assert_eq!(slab.free_count(), 4);

        let h1 = slab.alloc(42u32).unwrap();
        assert_eq!(slab.allocated(), 1);
        assert_eq!(slab.free_count(), 3);
        assert_eq!(*slab.get(h1).unwrap(), 42u32);

        slab.free(h1);
        assert_eq!(slab.allocated(), 0);
        assert_eq!(slab.free_count(), 4);
    }

    #[test]
    fn use_after_free_detection() {
        let mut slab: Slab<u32> = Slab::new(4);
        let h = slab.alloc(99u32).unwrap();
        slab.free(h);
        // Handle is now stale — get must return None.
        assert!(slab.get(h).is_none());
        assert!(slab.get_mut(h).is_none());
    }

    #[test]
    fn generation_increments_on_free() {
        let mut slab: Slab<u32> = Slab::new(2);
        let h1 = slab.alloc(1u32).unwrap();
        assert_eq!(h1.generation, 0);
        slab.free(h1);

        // Re-allocate the same index — generation should be 1.
        let h2 = slab.alloc(2u32).unwrap();
        if h2.index == h1.index {
            assert_eq!(h2.generation, 1);
        }
    }

    #[test]
    fn pool_exhaustion_returns_none() {
        let mut slab: Slab<u32> = Slab::new(2);
        let _h1 = slab.alloc(1u32).unwrap();
        let _h2 = slab.alloc(2u32).unwrap();
        let h3 = slab.alloc(3u32);
        assert!(h3.is_none(), "exhausted pool must return None");
    }

    // ── MessageBuffer tests ───────────────────────────────────────────────────

    #[test]
    fn message_buffer_write_and_read() {
        let mut buf = MessageBuffer::new(16);
        let written = buf.write(b"hello");
        assert_eq!(written, 5);
        assert_eq!(buf.read(), b"hello");
    }

    #[test]
    fn message_buffer_caps_at_capacity() {
        let mut buf = MessageBuffer::new(4);
        let written = buf.write(b"toolong");
        assert_eq!(written, 4);
        assert_eq!(buf.read(), b"tool");
    }

    #[test]
    fn message_buffer_clear() {
        let mut buf = MessageBuffer::new(8);
        buf.write(b"abc");
        buf.clear();
        assert_eq!(buf.len, 0);
        assert_eq!(buf.read(), b"");
    }

    // ── MessagePool tests ─────────────────────────────────────────────────────

    #[test]
    fn message_pool_acquire_release() {
        let mut pool = MessagePool::new(4, 64);
        let h = pool.acquire().unwrap();
        assert_eq!(pool.allocated(), 1);
        {
            let buf = pool.get_buffer(h).unwrap();
            buf.write(b"data");
        }
        pool.release(h);
        assert_eq!(pool.allocated(), 0);
        // Buffer cleared on release — handle now stale.
        assert!(pool.get_buffer(h).is_none());
    }

    #[test]
    fn message_pool_exhaustion() {
        let mut pool = MessagePool::new(2, 32);
        let _h1 = pool.acquire().unwrap();
        let _h2 = pool.acquire().unwrap();
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn message_pool_stale_handle_after_release() {
        let mut pool = MessagePool::new(2, 32);
        let h = pool.acquire().unwrap();
        pool.release(h);
        assert!(pool.get_buffer(h).is_none());
    }
}
