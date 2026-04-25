import re
import gymnasium as gym
import numpy as np

# Import AST definitions
from pyast_config import (
    GymEnvironmentConfig, DefinitionsConfig, VariablesConfig, ProblemConfig,
    ConstantDefinitionConfig, FunctionDefinitionConfig, PredicateDefinitionConfig,
    HybridProgramConfig,
    ControlFuncConfig, StepFuncConfig, ODEConfig, EvolutionDomainConfig,
    VariableDeclarationConfig, PreconditionConfig, PostconditionConfig,
    LoopExecutionConfig, HybridProgramStepConfig,
    ActionType, ChoiceActionConfig, SequenceActionConfig,
    DeterministicAssignmentActionConfig, NondeterministicAssignmentActionConfig,
    ActionConstraintConfig, CallHybridProgramConfig
)

# Import the main generator function
from gym_generator.main_generator import generate_gym_environment_code

def get_example_gym_config() -> GymEnvironmentConfig:
    """
    Creates and returns a dummy GymEnvironmentConfig based on the example PyAST structure.
    This mimics the output of convert_keymaerax_ast_to_intermediate_pyast.
    """
    
    # Constants
    # Set ep_const.value to None to demonstrate Z3 solving for it
    ep_const = ConstantDefinitionConfig(name='ep', type='Real', value=None) 
    b_const = ConstantDefinitionConfig(name='b', type='Real', value='0.5') # Set a value for testing
    A_const = ConstantDefinitionConfig(name='A', type='Real', value='1.0') # Set a value for testing
    W_const = ConstantDefinitionConfig(name='W', type='Real', value='1.0') # Set a value for testing
    V_const = ConstantDefinitionConfig(name='V', type='Real', value='1.0') # Added V constant for testing
    # FIX: Set Ud to a value < 1.0 for typical problem interpretation
    Ud_const = ConstantDefinitionConfig(name='Ud', type='Real', value='0.1') # Added Ud constant for testing

    # Functions
    # Bodies are now Python-compatible and use self.constants/self.functions
    stopDist_func = FunctionDefinitionConfig(name='stopDist', return_type='Real', parameters=[('Real', 'v')], body='((1+Ud) * v^2 / (2*b))') # Use ^ for power
    accelComp_func = FunctionDefinitionConfig(name='accelComp', return_type='Real', parameters=[('Real', 'v')], body='((1+Ud) * (A/b + 1) * (A/2 * ep^2 + ep*v))') # Use ^ for power
    admissibleSeparation_func = FunctionDefinitionConfig(name='admissibleSeparation', return_type='Real', parameters=[('Real', 'v')], body='stopDist(v) + accelComp(v)')

    # Predicates
    # Note: 'isWellformedDir' and 'initialState' use '=', which will now be approx.
    # Changed isWellformedDir to use equality for testing the fix
    isWellformedDir_pred = PredicateDefinitionConfig(name='isWellformedDir', parameters=[('Real', 'dx'), ('Real', 'dy')], body='dx^2 + dy^2 = 1', metadata={'type': 'progress', 'shape': '-((dx^2 + dy^2) - 1)^2', 'on_success': 'continue:status=none', 'on_failure': 'continue:status=none'})
    # FIX: Corrected bounds predicate to match 0 <= Ud < 1
    bounds_pred = PredicateDefinitionConfig(name='bounds', parameters=[], body='A >= 0\n    & b > 0\n    & ep > 0\n    & V >= 0\n    & 0<=Ud\n    & Ud < 1') # Use & for AND, added V, Ud bounds
    initialState_pred = PredicateDefinitionConfig(name='initialState', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='v = 0 & (x-xo)^2 + (y-yo)^2 > 0 & isWellformedDir(dx,dy)') # Use &
    
    # Added metadata for safety predicates
    isPhysicallyValid_pred = PredicateDefinitionConfig(name='isPhysicallyValid', parameters=[('Real', 'v'), ('Real', 'r')], body='v >= 0 & r != 0', metadata={'type': 'safety', 'penalty': '-500', 'on_failure': 'truncate:status=failure', 'on_success': 'continue:status=none'})
    isSafe_pred = PredicateDefinitionConfig(name='isSafe', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'xo'), ('Real', 'yo')], body='abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v)', metadata={'type': 'safety', 'penalty': '-1000', 'shape': '(abs(x-xo) - stopDist(v)) + (abs(y-yo) - stopDist(v))', 'on_failure': 'truncate:status=failure', 'on_success': 'continue:status=none'})
    isEfficient_pred = PredicateDefinitionConfig(name='isEfficient', parameters=[('Real', 'v')], body='v >= 0', metadata={'type': 'progress', 'shape': 'if v > 0 then 1/v else v', 'on_success': 'continue:status=none', 'on_failure': 'continue:status=none'})
    
    # Corrected loopinv body and parameters
    loopinv_pred = PredicateDefinitionConfig(name='loopinv', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo'), ('Real', 'r')], body='isPhysicallyValid(v, r) & isWellformedDir(dx, dy) & isSafe(x, y, v, xo, yo)')
    
    assumptions_pred = PredicateDefinitionConfig(name='assumptions', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'v'), ('Real', 'dx'), ('Real', 'dy'), ('Real', 'xo'), ('Real', 'yo')], body='bounds() & initialState(x,y,v,dx,dy,xo,yo)')
    
    ensure_pred = PredicateDefinitionConfig(name='ensure', parameters=[('Real', 'x'), ('Real', 'y'), ('Real', 'xo'), ('Real', 'yo')], body='(x - xo)^2 + (y - yo)^2 > 0', metadata={'type': 'progress', 'reward': '500', 'penalty': '-1000', 'on_success': 'terminate:status=success', 'on_failure': 'terminate:status=failure'})


    # Control Logic
    # This structure directly reflects the problem description's controlFunc
    # Choice 0: a := -b
    # Choice 1: v = 0; a := 0; w := 0
    # Choice 2: a := A; w := *; -W<=w & w<=W; r := *; xo := *; yo := *; r!=0 & r*w = v; abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparable(v)
    choice0_actions_branch0 = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='-b')
    ])
    choice0_actions_branch1 = SequenceActionConfig(actions=[
        ActionConstraintConfig(formula='v = 0'), # This '=' will now be approximate
        DeterministicAssignmentActionConfig(variable='a', value='0'),
        DeterministicAssignmentActionConfig(variable='w', value='0')
    ])
    choice0_actions_branch2 = SequenceActionConfig(actions=[
        DeterministicAssignmentActionConfig(variable='a', value='A'),
        NondeterministicAssignmentActionConfig(variable='w'),
        ActionConstraintConfig(formula='-W<=w & w<=W'), # Use &
        NondeterministicAssignmentActionConfig(variable='r'),
        NondeterministicAssignmentActionConfig(variable='xo'),
        NondeterministicAssignmentActionConfig(variable='yo'), # FIX: Corrected class name here
        ActionConstraintConfig(formula='r!=0 & r*w = v'), # This '=' will now be approximate
        ActionConstraintConfig(formula='abs(x-xo) > admissibleSeparation(v) | abs(y-yo) > admissibleSeparation(v)') # Use |
    ])

    control_func_action_config = ChoiceActionConfig(level_name='ChoiceLevel_0', choices=[
        choice0_actions_branch0, 
        choice0_actions_branch1,
        choice0_actions_branch2
    ])
    control_func = ControlFuncConfig(name='controlFunc', action_space_config=control_func_action_config)


    # stepFunc ODEs and Evolution Domain
    step_func_odes = [
        ODEConfig(variable='x', equation='v * dx'),
        ODEConfig(variable='y', equation='v * dy'),
        ODEConfig(variable='v', equation='a'),
        ODEConfig(variable='dx', equation='-w * dy'),
        ODEConfig(variable='dy', equation='w * dx'),
        ODEConfig(variable='w', equation='a/r'), # Corrected from '0'
        ODEConfig(variable='t', equation='1')
    ]
    step_func_evolution_domain = EvolutionDomainConfig(constraints=['t <= ep', 'v >= 0'])
    step_func = StepFuncConfig(name='stepFunc', odes=step_func_odes, evolution_domain=step_func_evolution_domain, duration='ep')

    # DefinitionsConfig
    definitions_config = DefinitionsConfig(
        constants=[ep_const, b_const, A_const, W_const, V_const, Ud_const], # Added V, Ud_const
        functions=[stopDist_func, accelComp_func, admissibleSeparation_func],
        predicates=[isWellformedDir_pred, isPhysicallyValid_pred, isSafe_pred, isEfficient_pred, bounds_pred, initialState_pred, loopinv_pred, assumptions_pred, ensure_pred],
        hybrid_programs=[control_func, step_func]
    )

    # VariablesConfig
    state_vars = [
        VariableDeclarationConfig(name='x', type='Real'),
        VariableDeclarationConfig(name='y', type='Real'),
        VariableDeclarationConfig(name='v', type='Real'),
        VariableDeclarationConfig(name='a', type='Real'),
        VariableDeclarationConfig(name='dx', type='Real'),
        VariableDeclarationConfig(name='dy', type='Real'),
        VariableDeclarationConfig(name='w', type='Real'),
        VariableDeclarationConfig(name='r', type='Real'),
        VariableDeclarationConfig(name='xo', type='Real'),
        VariableDeclarationConfig(name='yo', type='Real'),
        VariableDeclarationConfig(name='t', type='Real')
    ]
    variables_config = VariablesConfig(state_variables=state_vars)

    # ProblemConfig
    precondition = PreconditionConfig(formula='assumptions(x, y, v, dx, dy, xo, yo)')
    postcondition = PostconditionConfig(formula='ensure(x, y, xo, yo)') # Now calls the ensure predicate
    
    # Loop body: { { controlFunc; t := 0; } { stepFunc;@invariant(...) } }
    loop_body_actions = SequenceActionConfig(actions=[
        SequenceActionConfig(actions=[
            CallHybridProgramConfig(hp_name='controlFunc'),
            DeterministicAssignmentActionConfig(variable='t', value='0')
        ]),
        HybridProgramStepConfig(hp_name='stepFunc', invariant_formula='t>=0 & isWellformedDir(dx, dy) & (v>0 -> (abs(x-xo) > stopDist(v) | abs(y-yo) > stopDist(v)))') # Corrected invariant for example
    ])
    loop_execution = LoopExecutionConfig(
        iterations='inf',
        loop_invariant_formula='loopinv(x, y, v, dx, dy, xo, yo, r)', # Corrected loopinv parameters
        loop_body=loop_body_actions
    )
    problem_config = ProblemConfig(
        precondition=precondition,
        loop_execution=loop_execution,
        postcondition=postcondition
    )

    # Full GymEnvironmentConfig
    # This example env_name will now be sanitized:
    gym_config_example = GymEnvironmentConfig(
        env_name='IJRR17_Corollary_1_2__Static_safety_with_margin_for_imperfect_trajectories', # Original name
        definitions=definitions_config,
        variables=variables_config,
        problem=problem_config
    )
    return gym_config_example


if __name__ == "__main__":
    gym_config_example = get_example_gym_config()

    # Generate the code
    generated_code = generate_gym_environment_code(gym_config_example)

    # Print the generated code
    print("--- Generated Gymnasium Environment Code ---")
    print(generated_code)

    # Optionally, save to a file
    output_filename = "generated_kx_env.py"
    with open(output_filename, "w") as f:
        f.write(generated_code)
    print(f"\nGenerated Gymnasium environment code saved to {output_filename}")

    print("\n--- Example Usage Instructions ---")
    
    # Update the example usage to reflect the sanitized class name
    sanitized_example_env_name = re.sub(r'[^a-zA-Z0-9_]', '_', gym_config_example.env_name)
    sanitized_example_env_name = re.sub(r'_+', '_', sanitized_example_env_name).strip('_')
    if sanitized_example_env_name and sanitized_example_env_name[0].isdigit():
        sanitized_example_env_name = '_' + sanitized_example_env_name
    if not sanitized_example_env_name:
        sanitized_example_env_name = "GeneratedEnv"

    print(f"""
To use the generated environment:

1.  Make sure `gymnasium`, `numpy`, `scipy`, and `z3-solver` are installed (`pip install gymnasium numpy scipy z3-solver`).
2.  **CRITICAL:** Save the generated code above as `{output_filename}`. **Ensure you completely replace your existing generator file with this one.**
3.  Ensure `pyast_config.py` is in the same directory.
4.  You can then import and use the generated environment in another Python script or interactive session:

```python
import gymnasium as gym
from generated_kx_env import {sanitized_example_env_name} 
import numpy as np

# Instantiate the environment
# The __init__ method now calls reset(), so the environment starts in a valid state.
env = {sanitized_example_env_name}()

print("\\n--- Initial state after __init__ (from first reset call) ---")
# The first reset call is done internally by __init__
observation, info = env._get_obs(), env._get_info() # Manually get obs and info as reset() was called internally
print("Initial observation:", observation)
print("Initial state (info):", info)
if env.last_failing_precondition_formula:
    print(f"Precondition not fully satisfied: {{env.last_failing_precondition_formula}}")
print(f"Constants after first reset: {{env.constants}}") # Check if ep was solved by Z3

print("\\n--- Second Reset (should be different and valid) ---")
observation, info = env.reset() # Explicitly call reset again for a new episode
print("Initial observation:", observation)
print("Initial state (info):", info)
if env.last_failing_precondition_formula:
    print(f"Precondition not fully satisfied: {{env.last_failing_precondition_formula}}")
print(f"Constants after second reset: {{env.constants}}") # Check if ep was solved by Z3

# Example 1: Choose the first branch of ChoiceLevel_0 (a := -b)
print("\\nTaking 5 steps with action_branch_0 (a := -b):")
env.reset() # Reset for a fresh start for this example
for i in range(5):
    print(f"Step {{i+1}} (before action): State={{env.state}}")
    action_to_take = {{
        'choice_ChoiceLevel_0': 0, # Select the first branch (a := -b)
    }}
    observation, reward, terminated, truncated, info = env.step(action_to_take)
    print(f"Step {{i+1}}: Obs={{observation}}, Reward={{reward:.2f}}, Terminated={{terminated}}, Truncated={{truncated}}")
    if terminated or truncated:
        print("Episode finished.")
        break
env.close()

# Example 2: Reset and choose the third branch of ChoiceLevel_0
# (a := A; w := *; etc.)
print("\\nResetting environment and taking 5 steps with action_branch_2 (a := A; w := *; etc.):")
env = {sanitized_example_env_name}() # Re-instantiate for a fresh start
observation, info = env.reset()

# Define action for ChoiceLevel_0 branch 2 (index 2)
# It's crucial that r != 0 and r*w = v for the constraint to pass.
# Let's assume v is initially 0.0 (from initialState).
# If w = 0.0, then r*w = 0.0, which matches v=0.0.
action_branch_2 = {{
    'choice_ChoiceLevel_0': 2, # Select the third branch (a := A; w := *; etc.)
    'assign_w': 0.0,           # Example value for w (within -W to W, i.e., -1.0 to 1.0)
    'assign_r': 1.0,           # Example value for r (r!=0)
    'assign_xo': 5.0,          # Example value for xo
    'assign_yo': 0.0,          # Example value for yo
}}

for i in range(5):
    print(f"Step {{i+1}} (before action): State={{env.state}}")
    observation, reward, terminated, truncated, info = env.step(action_branch_2)
    print(f"Step {{i+1}}: Obs={{observation}}, Reward={{reward:.2f}}, Terminated={{terminated}}, Truncated={{truncated}}")
    if terminated or truncated:
        print("Episode finished.")
        break
env.close()
""")