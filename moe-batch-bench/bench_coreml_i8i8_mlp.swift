import CoreML
import Foundation

struct Args {
    var modelPath: String = ""
    var warmup: Int = 3
    var iters: Int = 20
    var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
}

func usage(_ prog: String) -> Never {
    fputs("usage: \(prog) model.mlpackage [--warmup N] [--iters N] [--compute-units cpu_ane|all|cpu_gpu|cpu]\n", stderr)
    exit(2)
}

func parseArgs() -> Args {
    var args = Args()
    let argv = CommandLine.arguments
    if argv.count < 2 {
        usage(argv[0])
    }
    args.modelPath = argv[1]
    var i = 2
    while i < argv.count {
        switch argv[i] {
        case "--warmup":
            i += 1
            if i >= argv.count || Int(argv[i]) == nil { usage(argv[0]) }
            args.warmup = Int(argv[i])!
        case "--iters":
            i += 1
            if i >= argv.count || Int(argv[i]) == nil { usage(argv[0]) }
            args.iters = Int(argv[i])!
        case "--compute-units":
            i += 1
            if i >= argv.count { usage(argv[0]) }
            switch argv[i] {
            case "cpu_ane": args.computeUnits = .cpuAndNeuralEngine
            case "all": args.computeUnits = .all
            case "cpu_gpu": args.computeUnits = .cpuAndGPU
            case "cpu": args.computeUnits = .cpuOnly
            default: usage(argv[0])
            }
        default:
            usage(argv[0])
        }
        i += 1
    }
    if args.warmup < 0 || args.iters <= 0 {
        usage(argv[0])
    }
    return args
}

func product(_ shape: [NSNumber]) -> Int {
    shape.reduce(1) { $0 * $1.intValue }
}

func fillInt8(_ array: MLMultiArray, seed: UInt32) {
    let n = array.count
    let ptr = array.dataPointer.bindMemory(to: Int8.self, capacity: n)
    var s = seed
    for idx in 0..<n {
        s = s &* 1_664_525 &+ 1_013_904_223
        ptr[idx] = Int8(Int(s % 255) - 127)
    }
}

func fillInt32(_ array: MLMultiArray, seed: UInt32) {
    let n = array.count
    let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: n)
    var s = seed
    for idx in 0..<n {
        s = s &* 1_664_525 &+ 1_013_904_223
        ptr[idx] = Int32(Int(s % 255) - 127)
    }
}

func makeInputArray(name: String, constraint: MLMultiArrayConstraint, seed: UInt32) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
    switch constraint.dataType {
    case .int8:
        fillInt8(array, seed: seed)
    case .int32:
        fillInt32(array, seed: seed)
    default:
        throw NSError(
            domain: "bench_coreml_i8i8_mlp",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "unsupported input dtype for \(name): \(constraint.dataType)"]
        )
    }
    return array
}

func inferKindAndShape(_ inputs: [String: MLFeatureDescription]) -> (String, Int, Int, Int) {
    if let wg = inputs["Wgq"]?.multiArrayConstraint,
       let x = inputs["Xq"]?.multiArrayConstraint {
        let h = wg.shape[0].intValue
        let interm = wg.shape[1].intValue
        let b = x.shape[0].intValue
        return (inputs["Wdq"] == nil ? "gateup" : "full", h, interm, b)
    }
    if let wd = inputs["Wdq"]?.multiArrayConstraint,
       let hidden = inputs["hidden_q"]?.multiArrayConstraint {
        let interm = wd.shape[0].intValue
        let h = wd.shape[1].intValue
        let b = hidden.shape[0].intValue
        return ("down", h, interm, b)
    }
    return ("unknown", 0, 0, 0)
}

func effectiveTFLOPs(kind: String, h: Int, i: Int, b: Int, ms: Double) -> Double {
    let matmulFactor: Double
    switch kind {
    case "gateup": matmulFactor = 4.0
    case "down": matmulFactor = 2.0
    default: matmulFactor = 6.0
    }
    let flops = matmulFactor * Double(b) * Double(h) * Double(i)
    return flops / (ms / 1000.0) / 1.0e12
}

func main() throws {
    let args = parseArgs()
    let config = MLModelConfiguration()
    config.computeUnits = args.computeUnits
    let modelURL = URL(fileURLWithPath: args.modelPath)
    let loadURL: URL
    if modelURL.pathExtension == "mlmodelc" {
        loadURL = modelURL
    } else {
        loadURL = try MLModel.compileModel(at: modelURL)
    }
    let model = try MLModel(contentsOf: loadURL, configuration: config)
    let inputDesc = model.modelDescription.inputDescriptionsByName
    let (kind, h, interm, b) = inferKindAndShape(inputDesc)

    var dict: [String: MLFeatureValue] = [:]
    var dtypeReport: [String] = []
    var seed: UInt32 = 0x1234
    for name in inputDesc.keys.sorted() {
        guard let constraint = inputDesc[name]?.multiArrayConstraint else {
            continue
        }
        let array = try makeInputArray(name: name, constraint: constraint, seed: seed)
        seed &+= 0x1111
        dict[name] = MLFeatureValue(multiArray: array)
        dtypeReport.append("\(name): \(constraint.dataType)")
    }
    let provider = try MLDictionaryFeatureProvider(dictionary: dict)

    for _ in 0..<args.warmup {
        _ = try model.prediction(from: provider)
    }

    var times: [Double] = []
    var outputReport: [String] = []
    for iter in 0..<args.iters {
        let start = DispatchTime.now().uptimeNanoseconds
        let out = try model.prediction(from: provider)
        let end = DispatchTime.now().uptimeNanoseconds
        times.append(Double(end - start) / 1.0e6)
        if iter == 0 {
            for name in out.featureNames.sorted() {
                if let arr = out.featureValue(for: name)?.multiArrayValue {
                    outputReport.append("\(name): \(arr.shape)")
                }
            }
        }
    }
    let mean = times.reduce(0, +) / Double(times.count)
    let sorted = times.sorted()
    let median = sorted[sorted.count / 2]
    let minValue = sorted.first ?? mean
    let tflops = effectiveTFLOPs(kind: kind, h: h, i: interm, b: b, ms: mean)
    let dtypeText = dtypeReport.joined(separator: ", ")
    let outputText = outputReport.joined(separator: ", ")
    print("model=\(args.modelPath) kind=\(kind) H=\(h) I=\(interm) B=\(b) warmup=\(args.warmup) iters=\(args.iters)")
    print("input_dtypes={\(dtypeText)}")
    print("outputs={\(outputText)}")
    print(String(format: "mean_ms=%.6f median_ms=%.6f min_ms=%.6f effective_TFLOPs=%.6f", mean, median, minValue, tflops))
}

do {
    try main()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
