from parol6 import RobotClient

HOST = "127.0.0.1"
PORT = 5001


def main() -> None:
    with RobotClient(host=HOST, port=PORT, timeout=2.0) as client:
        ready = client.wait_for_server_ready(timeout=5.0)
        if not ready:
            print("Server not ready, aborting.")
            raise SystemExit(1)

        print("Sending home command...")
        success = client.home()
        print(f"Home command result: {success}")

        print("Angles after homing:", client.get_angles())

    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
