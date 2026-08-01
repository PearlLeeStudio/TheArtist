// Native MIDI 1.0 input for the local browser bridge. No MIDI output or file access.
import Foundation
import CoreMIDI

let outputLock = NSLock()
func emit(_ value: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else { return }
    outputLock.lock()
    FileHandle.standardOutput.write(data + Data([10]))
    outputLock.unlock()
}

final class Decoder {
    var status: UInt8 = 0
    var bytes: [UInt8] = []
    func read(_ data: [UInt8], port: String) {
        for byte in data {
            if byte >= 0xF8 { continue } // Interleaved realtime messages do not clear running status.
            if byte >= 0x80 {
                bytes = []
                status = byte < 0xF0 ? byte : 0 // Ignore SysEx and system messages.
                continue
            }
            if status == 0 { continue }
            bytes.append(byte)
            let kind = status & 0xF0
            let count = (kind == 0xC0 || kind == 0xD0) ? 1 : 2
            if bytes.count == count {
                if kind == 0x80 || kind == 0x90 || kind == 0xB0 {
                    emit(["type": "midi", "port": port, "data": [Int(status)] + bytes.map(Int.init)])
                }
                bytes = []
            }
        }
    }
}

if CommandLine.arguments.contains("--self-test") {
    let decoder = Decoder()
    decoder.read([0x90, 60, 100, 64, 100, 0xF8, 67, 100], port: "self-test")
    decoder.read([0x80, 60, 0, 64, 0, 67, 0, 0xB0, 123, 0], port: "self-test")
    decoder.read([0xF0, 1, 2, 3, 0xF7], port: "self-test")
    exit(0)
}

var client = MIDIClientRef()
var inputPort = MIDIPortRef()
var attached = Set<MIDIEndpointRef>()
var decoders: [UInt: Decoder] = [:]
let decoderLock = NSLock()

func refresh() {
    let sources = Set((0..<MIDIGetNumberOfSources()).map { MIDIGetSource($0) }.filter { $0 != 0 })
    for source in attached.subtracting(sources) {
        MIDIPortDisconnectSource(inputPort, source)
        decoderLock.lock(); decoders.removeValue(forKey: UInt(source)); decoderLock.unlock()
    }
    var connected = attached.intersection(sources)
    for source in sources.subtracting(attached) {
        if MIDIPortConnectSource(inputPort, source, UnsafeMutableRawPointer(bitPattern: UInt(source))) == noErr {
            connected.insert(source)
        }
    }
    attached = connected
    let devices = connected.sorted().map { source -> [String: String] in
        var name: Unmanaged<CFString>?
        let status = MIDIObjectGetStringProperty(source, kMIDIPropertyDisplayName, &name)
        return ["id": String(source), "name": status == noErr ? (name?.takeRetainedValue() as String? ?? "MIDI input") : "MIDI input"]
    }
    emit(["type": "state", "devices": devices])
}

let status = MIDIClientCreateWithBlock("TheArtist MIDI Input" as CFString, &client) { _ in
    DispatchQueue.main.async { if inputPort != 0 { refresh() } }
}
guard status == noErr else {
    emit(["type": "error", "message": "CoreMIDI client could not start", "status": status]); exit(1)
}
let portStatus = MIDIInputPortCreateWithBlock(client, "TheArtist browser input" as CFString, &inputPort) { list, context in
    let source = UInt(bitPattern: context)
    decoderLock.lock()
    let decoder = decoders[source] ?? Decoder()
    decoders[source] = decoder
    let offset = MemoryLayout<MIDIPacketList>.offset(of: \.packet)!
    var packet = UnsafeRawPointer(list).advanced(by: offset).assumingMemoryBound(to: MIDIPacket.self)
    for _ in 0..<list.pointee.numPackets {
        let bytes = UnsafeRawPointer(packet).advanced(by: MemoryLayout<MIDIPacket>.offset(of: \.data)!).assumingMemoryBound(to: UInt8.self)
        decoder.read(Array(UnsafeBufferPointer(start: bytes, count: Int(packet.pointee.length))), port: String(source))
        packet = UnsafePointer(MIDIPacketNext(packet))
    }
    decoderLock.unlock()
}
guard portStatus == noErr else {
    emit(["type": "error", "message": "CoreMIDI input port could not open", "status": portStatus]); exit(1)
}
refresh()
if CommandLine.arguments.contains("--probe") {
    MIDIPortDispose(inputPort); MIDIClientDispose(client); exit(0)
}
RunLoop.main.run()
