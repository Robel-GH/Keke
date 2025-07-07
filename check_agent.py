import os
import importlib.util
import sys

def check_agent(agent_name="hybrid_astar_AGENT"):
    agent_path = f"agents/{agent_name}.py"
    
    print(f"Checking agent: {agent_name}")
    print("-" * 50)
    
    # Check if file exists
    if not os.path.exists(agent_path):
        print(f"❌ Error: Agent file not found at {agent_path}")
        return False
        
    try:
        # Try to load the module
        spec = importlib.util.spec_from_file_location(agent_name, agent_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check for the agent class
        class_name = "".join(word.capitalize() for word in agent_name.split('_'))
        if not hasattr(module, class_name):
            print(f"❌ Error: Could not find {class_name} class in {agent_path}")
            return False
            
        agent_class = getattr(module, class_name)
        
        # Try to instantiate the agent
        agent = agent_class()
        print("✅ Agent loaded successfully!")
        print(f"Class name: {class_name}")
        print(f"Methods available: {[method for method in dir(agent) if not method.startswith('_')]}")
        
        # Check for required methods
        if not hasattr(agent, 'search'):
            print("❌ Error: Agent missing required 'search' method")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading agent: {str(e)}")
        return False

if __name__ == "__main__":
    check_agent()