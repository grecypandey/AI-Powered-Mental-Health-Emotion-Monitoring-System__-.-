import asyncio
from bleak import BleakScanner, BleakClient

# UUID for standard Heart Rate Measurement
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


async def scan_and_connect():
    print("🔍 Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover()

    if not devices:
        print("❌ No Bluetooth devices found.")
        return

    # List available devices
    for i, d in enumerate(devices):
        print(f"[{i}] {d.name} ({d.address})")

    # Ask user to pick one
    choice = int(input("👉 Select device number to connect: "))
    if choice < 0 or choice >= len(devices):
        print("❌ Invalid choice.")
        return

    selected = devices[choice]
    print(f"✅ Connecting to {selected.name} ({selected.address})")

    async with BleakClient(selected.address) as client:
        print("📡 Connected! Waiting for heart rate data...")

        def callback(sender, data):
            # Heart rate is usually in the 2nd byte
            hr_value = int(data[1])
            print(f"❤️ Heart Rate: {hr_value} BPM")

        # Subscribe to notifications
        await client.start_notify(HR_UUID, callback)

        # Keep listening
        await asyncio.sleep(30)  # read for 30 seconds
        await client.stop_notify(HR_UUID)
        print("🔌 Disconnected.")


if __name__ == "__main__":
    asyncio.run(scan_and_connect())
