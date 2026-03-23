// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "MetalBenchmark",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "MetalBenchmark",
            dependencies: [],
            path: "Sources/MetalBenchmark"
        )
    ]
)
