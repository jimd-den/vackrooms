//! Exact in-flight request bookkeeping for the generation-worker pool.
//!
//! Browser callbacks can arrive after a worker has failed or after the engine
//! has already retried a chunk. This small ledger gives every callback one
//! deterministic operation: finish the exact request it echoes, fail the
//! oldest request after a malformed reply, or drain a failed worker.

use std::collections::VecDeque;

use crate::application::ports::ChunkRequest;

#[derive(Debug)]
struct WorkerRequests {
    accepting_work: bool,
    in_flight: VecDeque<ChunkRequest>,
}

/// Per-worker request ownership. Request identity is always checked through
/// [`ChunkRequest::is_same_request`], never by chunk coordinates alone.
#[derive(Debug)]
pub(crate) struct GenerationWorkerRequests {
    workers: Vec<WorkerRequests>,
}

impl GenerationWorkerRequests {
    pub(crate) fn new(worker_count: usize) -> Self {
        assert!(
            worker_count > 0,
            "a worker ledger needs at least one worker"
        );
        Self {
            workers: (0..worker_count)
                .map(|_| WorkerRequests {
                    accepting_work: true,
                    in_flight: VecDeque::new(),
                })
                .collect(),
        }
    }

    /// Finds the next healthy worker, wrapping once from `start`.
    pub(crate) fn next_accepting(&self, start: usize) -> Option<usize> {
        (0..self.workers.len())
            .map(|offset| (start + offset) % self.workers.len())
            .find(|&index| self.workers[index].accepting_work)
    }

    pub(crate) fn assign(&mut self, worker_index: usize, request: ChunkRequest) {
        let worker = &mut self.workers[worker_index];
        debug_assert!(worker.accepting_work);
        worker.in_flight.push_back(request);
    }

    /// Retires only the exact request echoed by a well-formed worker reply.
    /// Unknown and stale replies leave all current work untouched.
    pub(crate) fn finish_exact(&mut self, worker_index: usize, echoed: &ChunkRequest) -> bool {
        self.take_exact(worker_index, echoed).is_some()
    }

    /// Retires an exact request reported as failed by the worker protocol.
    pub(crate) fn fail_exact(
        &mut self,
        worker_index: usize,
        echoed: &ChunkRequest,
    ) -> Option<ChunkRequest> {
        self.take_exact(worker_index, echoed)
    }

    /// A worker processes its queue serially, so a malformed response belongs
    /// to its oldest outstanding request even when its envelope cannot be
    /// decoded far enough to recover an id.
    pub(crate) fn fail_oldest(&mut self, worker_index: usize) -> Option<ChunkRequest> {
        self.workers[worker_index].in_flight.pop_front()
    }

    /// Stops scheduling to a broken worker and returns all work the engine
    /// must retry. Calling this more than once is harmless.
    pub(crate) fn disable_and_drain(&mut self, worker_index: usize) -> Vec<ChunkRequest> {
        let worker = &mut self.workers[worker_index];
        worker.accepting_work = false;
        worker.in_flight.drain(..).collect()
    }

    fn take_exact(&mut self, worker_index: usize, echoed: &ChunkRequest) -> Option<ChunkRequest> {
        let queue = &mut self.workers[worker_index].in_flight;
        let position = queue
            .iter()
            .position(|request| request.is_same_request(echoed))?;
        queue.remove(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vackrooms::domain::entities::anomaly::RealitySnapshot;

    fn request(id: u32, x: f32) -> ChunkRequest {
        ChunkRequest {
            request_id: id,
            origin_x: x,
            origin_z: -4.0,
            level: 0,
            lod: 1,
            reality: RealitySnapshot::default(),
        }
    }

    #[test]
    fn scheduling_wraps_and_skips_disabled_workers() {
        let mut requests = GenerationWorkerRequests::new(3);
        assert_eq!(requests.next_accepting(2), Some(2));
        requests.disable_and_drain(2);
        assert_eq!(requests.next_accepting(2), Some(0));
        requests.disable_and_drain(0);
        requests.disable_and_drain(1);
        assert_eq!(requests.next_accepting(0), None);
    }

    #[test]
    fn stale_reply_cannot_retire_current_work() {
        let mut requests = GenerationWorkerRequests::new(1);
        let current = request(8, 12.0);
        requests.assign(0, current.clone());

        let mut stale = current.clone();
        stale.request_id = 7;
        assert!(!requests.finish_exact(0, &stale));
        assert!(requests.finish_exact(0, &current));
    }

    #[test]
    fn malformed_reply_fails_only_the_oldest_serial_request() {
        let mut requests = GenerationWorkerRequests::new(1);
        let first = request(1, 0.0);
        let second = request(2, 16.0);
        requests.assign(0, first.clone());
        requests.assign(0, second.clone());

        assert_eq!(requests.fail_oldest(0), Some(first));
        assert!(requests.finish_exact(0, &second));
    }

    #[test]
    fn worker_failure_drains_its_work_without_affecting_other_workers() {
        let mut requests = GenerationWorkerRequests::new(2);
        let first = request(1, 0.0);
        let second = request(2, 16.0);
        requests.assign(0, first.clone());
        requests.assign(1, second.clone());

        assert_eq!(requests.disable_and_drain(0), vec![first]);
        assert_eq!(requests.next_accepting(0), Some(1));
        assert!(requests.finish_exact(1, &second));
        assert!(requests.disable_and_drain(0).is_empty());
    }

    #[test]
    fn explicit_failure_requires_the_exact_request_identity() {
        let mut requests = GenerationWorkerRequests::new(1);
        let current = request(5, 0.0);
        requests.assign(0, current.clone());
        let mut wrong_identity = current.clone();
        wrong_identity.lod = 0;

        assert!(requests.fail_exact(0, &wrong_identity).is_none());
        assert_eq!(requests.fail_exact(0, &current), Some(current));
    }
}
