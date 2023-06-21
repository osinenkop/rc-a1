import traceback
import sys

import system


def launch():

    try:
        system = system.A1system(
            A1_topic_name_res="/robot_1/calf/action",
            dim_state=12,
            dim_input=12,
        )
        return 0
    except Exception:
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])


if __name__ == "__main__":
    job_results = launch()
