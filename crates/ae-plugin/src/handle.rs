use core::slice;
use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::{transmute, MaybeUninit},
    ptr::NonNull,
};

use after_effects::{suites, sys::Ptr, Error};

pub struct SliceHandle<T> {
    handle: NonNull<Ptr>,
    _ty: PhantomData<T>,
    len: usize,
    suite: suites::Handle,
}

impl<T> SliceHandle<T> {
    pub fn new_uninit(len: usize) -> Result<SliceHandle<MaybeUninit<T>>, Error> {
        let handle_suite = suites::Handle::new()?;
        let handle = NonNull::new(handle_suite.new_handle((len * std::mem::size_of::<T>()) as u64))
            .ok_or(Error::OutOfMemory)?;

        Ok(SliceHandle {
            handle: handle as _,
            _ty: PhantomData::default(),
            len,
            suite: handle_suite,
        })
    }

    pub fn lock(&mut self) -> Result<LockedHandle<'_, T>, Error> {
        let ptr = self.suite.lock_handle(self.handle.as_ptr());
        let data = NonNull::new(ptr as *mut T).ok_or(Error::BadCallbackParameter)?;

        Ok(LockedHandle { data, parent: self })
    }
}

impl<T: Copy> SliceHandle<T> {
    pub fn new(len: usize, value: T) -> Result<Self, Error> {
        let mut handle = Self::new_uninit(len)?;

        {
            let mut locked = handle.lock()?;
            let data: &mut [MaybeUninit<T>] = locked.borrow_mut();
            for i in data.iter_mut() {
                i.write(value);
            }
        }

        Ok(unsafe { transmute(handle) })
    }
}

pub struct LockedHandle<'a, T> {
    data: NonNull<T>,
    parent: &'a mut SliceHandle<T>,
}

impl<T> Borrow<[T]> for LockedHandle<'_, T> {
    fn borrow(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.parent.len) }
    }
}

impl<T> BorrowMut<[T]> for LockedHandle<'_, T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.parent.len) }
    }
}

impl<T> Drop for LockedHandle<'_, T> {
    fn drop(&mut self) {
        self.parent.suite.unlock_handle(self.parent.handle.as_ptr())
    }
}
