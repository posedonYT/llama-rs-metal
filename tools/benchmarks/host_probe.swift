import Foundation
let p = ProcessInfo.processInfo
let result: [String: Any] = [
    "thermal_state": p.thermalState.rawValue,
    "low_power_mode": p.isLowPowerModeEnabled,
    "physical_memory_bytes": p.physicalMemory,
    "os": p.operatingSystemVersionString,
    "uptime_seconds": p.systemUptime,
    "active_processors": p.activeProcessorCount
]
let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
