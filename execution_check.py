from execution import Execution
import json

# Create execution instance
executor = Execution(
    agent_path="agents/hybrid_astar_AGENT.py",
    level_path="json_levels/demo_LEVELS.json",
    iter_cap=100
)

# Enable detailed logging
executor.debug = True

# Run all levels and save results
results = executor.run_all_levels()

# Print detailed results
print("Execution Results:")
print("-----------------")
print(f"Total Levels: {len(results['levels'])}")
for level in results['levels']:
    print(f"\nLevel {level['id']}:")
    print(f"Status: {level['status']}")
    if 'solution' in level:
        print(f"Solution Length: {len(level['solution'])}")
        print(f"Solution Path: {level['solution']}")
    if 'error' in level:
        print(f"Error: {level['error']}")
    print(f"Time: {level.get('time', 'N/A')}s")

# Save results to file
with open('detailed_results.json', 'w') as f:
    json.dump(results, f, indent=2)