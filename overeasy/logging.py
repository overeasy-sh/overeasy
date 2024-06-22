import time
from tabulate import tabulate
from functools import wraps
from collections import defaultdict
from typing import Callable, Any, Dict

function_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "total_time": 0.0})

def log_time(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        if args and hasattr(args[0], '__class__'):
            name = f"{args[0].__class__.__name__}.{func.__name__}"
        else:
            name = func.__qualname__
        
        
        function_stats[name]["count"] += 1
        function_stats[name]["total_time"] += end_time - start_time

        return result
    return wrapper


def print_summary():
    total_run_time = sum(stats["total_time"] for stats in function_stats.values())
    table = []

    for func_name, stats in function_stats.items():\

        average_time = stats["total_time"] / stats["count"]

        if stats['total_time'] < 0.1:
            total_time_str = f"{average_time*1000:.2f}ms"
        elif stats['total_time'] < 60:
            total_time_str = f"{average_time:.2f}s"
        else:
            minutes, seconds = divmod(average_time, 60)
            total_time_str = f"{int(minutes)}m {seconds:.2f}s"
            
            
        proportion = (stats["total_time"] / total_run_time) * 100
        table.append([func_name, stats['count'], total_time_str, f"{proportion:.2f}%"])

    headers = ["Function Name", "Calls", "Average Time (s)", "Proportion of Total Runtime"]
    print(tabulate(table, headers=headers, tablefmt="grid"))