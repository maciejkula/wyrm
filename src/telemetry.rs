//! Telemetry tools.
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

lazy_static! {
    /// Holds telemetry objects.
    pub static ref TELEMETRY: Mutex<HashMap<&'static str, Timer>> = Mutex::new(HashMap::new());
}

/// Timer for telemetry.
#[derive(Default, Debug, Clone)]
pub struct Timer {
    elapsed: Duration,
    invocations: usize,
}

impl Timer {
    /// Add duration to a timer.
    pub(crate) fn add(&mut self, elapsed: Duration) {
        self.elapsed += elapsed;
        self.invocations += 1;
    }
}

/// Retrieve the telemetry data.
pub fn get_telemetry() -> Vec<(&'static str, Timer)> {
    let mut timers: Vec<(&'static str, Timer)> = TELEMETRY
        .lock()
        .unwrap()
        .iter()
        .map(|(x, y)| (x.clone(), y.clone()))
        .collect();

    timers.sort_by_key(|&(x, _)| x);

    timers
}

#[macro_export]
#[cfg(feature = "telemetry")]
macro_rules! measure {
    ($name:expr, $body:block) => {{
        use std::time::Instant;
        use telemetry;

        let start = Instant::now();
        let mut run = || $body;
        let result = run();
        let elapsed = start.elapsed();

        telemetry::TELEMETRY
            .lock()
            .unwrap()
            .entry($name)
            .or_insert(telemetry::Timer::default())
            .add(elapsed);

        result
    }};
}

#[macro_export]
#[cfg(not(feature = "telemetry"))]
macro_rules! measure {
    ($name:expr, $body:block) => {
        $body
    };
}
