import asyncio
import time
from typing import TypeVar

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional

T = TypeVar("T")

@dataclass
class RunState:
    command_line: List[str] = field(default_factory=list)
    durations: Dict[str, float] = field(default_factory=dict)
    
    
@dataclass
class FocusStep(Generic[T]):
    step: Callable[[T], Awaitable[None]]
    # A message to be displayed to the user at the beginning of the step
    user_message: Optional[str] = None
    # If the step can be skipped, you can provide the skipping logic here.
    should_run: Optional[Callable[[T], bool]] = None
    # Required for duration logging
    name: str = ""
    # Required for duration logging
    substep_durations: Optional[Dict[str, Any]] = None
    # Required for duration logging
    core: bool = False
    # Keep going if an error is encountered
    keep_going: bool = False

# a decorator factory
def focus_step(
    *,
    user_message: Optional[str] = None,
    should_run: Optional[Callable[[T], bool]] = None
) -> Callable[[Callable[[T], Awaitable[None]]], FocusStep[T]]:
    # first define a decorator
    def focus_step_decorator(func: Callable[[T], Awaitable[None]]) -> FocusStep[T]:
         step = FocusStep(
            step = func,
            user_message = user_message,
            should_run = should_run
        )
         return step
    return focus_step_decorator


@focus_step()
async def test1(run_state: RunState) -> None:
    print("test1 is running")
    await asyncio.sleep(5)
    print("test1 is done")
    
@focus_step()
async def test2(run_state: RunState) -> None:
    print("test2 is running")
    await asyncio.sleep(5)
    print("test2 is done")

@focus_step()
def test3(run_state: RunState) -> None:
    print("test3 is running")
    asyncio.sleep(2)
    print("test3 is done")
###########

async def perform_steps(
    steps: List[FocusStep[T]],
    run_state: T,
    *,
    durations: Dict[str, Any],
    # pyre-ignore [2] No better type give the python version
    logging_func: Callable[[Any], None],
) -> int:
    exit_code: int = 0
    durations["total"] = 0.0
    for step in steps:
        duration: float = time.time()
        # pyre-ignore [29]
        if step.should_run is not None and not step.should_run(run_state):
            continue
        if step.user_message is not None:
            logging_func(step.user_message)
        try:
            if step.core:
                # pyre-fixme[19]: Expected 1 positional argument.
                
                await step.step(run_state, step)
            else:
                # await step.step(run_state)
                print("hit!")
                task = asyncio.create_task(step.step(run_state))
                await task
        except (Exception, KeyboardInterrupt, SystemExit) as ex:
            print("exception: ", ex)
        duration = time.time() - duration
        name = step.name
        durations[name] = duration
    return exit_code


def focus_steps(  # noqa C901 No, you're too complex
    *steps: FocusStep[T],
) -> Callable[[T], Awaitable[int]]:
    async def inner(run_state: T, *, should_log: bool = False) -> int:
        durations: Dict[str, Any] = {}
        durations["total"] = 0
        run_state.durations = durations
        exit_code = await perform_steps(
            list(steps), run_state, durations=durations, logging_func=None
        )

        return exit_code

    return inner

def steps() -> Callable[[RunState], Awaitable[int]]:
    return focus_steps (
        test1,
        test2,
        # test3       
    )

def run(
    steps: Callable[[T], Awaitable[int]],
    run_state: T
) -> None:
     asyncio.run(steps(run_state))


def main():
    run(
        steps(), #steps() returns a coroutine - Callable[[T], Awaitable[int]]
        RunState()
    )
    
if __name__ == "__main__":
    main()