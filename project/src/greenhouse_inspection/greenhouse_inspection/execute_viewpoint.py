"""GPS-package entry point for the frozen, safety-checked flight executor."""

from greenhouse_inspection.viewpoint_execution import main as execute


def main(args=None):
    """Run the shared executor; its ``start_flight`` default remains false."""
    return execute(args=args)


if __name__ == "__main__":
    main()
