
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import sys
from scipy.integrate import solve_ivp
import sympy
import z3
from collections import OrderedDict, deque
import copy
import re
import math
import ast
from typing import Dict, Any, Optional, Union, List

from ast_config import (
    ASTNode, Term, Formula,
    Literal, VarRef, BinaryTermOp, UnaryTermOp, FunctionCall,
    BooleanLiteral, BinaryFormulaOp, UnaryFormulaOp, Comparison, PredicateCall,
    Assignment, NondetAssignment, Test, Sequence, Choice, Iteration,
    ODE, ODEEquation, HPBlock, ModalFormula, Quantifier, C_Formula,
    HybridProgram
)

# Embedded environment config (generator will replace the placeholder)
_ENV_CONFIG_JSON = '    {\n        "environment_name": "CruiseControlPPO",\n        "variables": [\n            {\n                "name": "x",\n                "type": "Real",\n                "lower_bound": "-500.0",\n                "upper_bound": "500.0",\n                "initial_value": null\n            },\n            {\n                "name": "t",\n                "type": "Real",\n                "lower_bound": "0.0",\n                "upper_bound": "2.5",\n                "initial_value": null\n            },\n            {\n                "name": "v",\n                "type": "Real",\n                "lower_bound": "-20.0",\n                "upper_bound": "40.0",\n                "initial_value": null\n            },\n            {\n                "name": "mode",\n                "type": "Real",\n                "lower_bound": "0.0",\n                "upper_bound": "5.0",\n                "initial_value": null\n            }\n        ],\n        "constants": [],\n        "function_definitions": [\n            {\n                "name": "accelerate",\n                "args": [],\n                "body": "0.0"\n            },\n            {\n                "name": "ebrakeact",\n                "args": [],\n                "body": "1.0"\n            },\n            {\n                "name": "ebrakefull",\n                "args": [],\n                "body": "2.0"\n            },\n            {\n                "name": "normalPI",\n                "args": [],\n                "body": "3.0"\n            },\n            {\n                "name": "sbrakeact",\n                "args": [],\n                "body": "4.0"\n            },\n            {\n                "name": "sbrakefull",\n                "args": [],\n                "body": "5.0"\n            },\n            {\n                "name": "cruiseSafe",\n                "args": [],\n                "body": "((((((((-(500.0) <= x) and (x <= 500.0)) and (-(20.0) <= v)) and (v <= 40.0)) and (0.0 <= t)) and (t <= 2.5)) and (0.0 <= mode)) and (mode <= 5.0))"\n            },\n            {\n                "name": "nearZeroSpeed",\n                "args": [],\n                "body": "(abs(v) <= 1.0)"\n            }\n        ],\n        "hp_definitions": [\n            {\n                "name": "initializeMode",\n                "args": [],\n                "body": "{(mode := self._ebrakeact_impl() ++ (mode := self._ebrakefull_impl() ++ (mode := self._sbrakeact_impl() ++ (mode := self._sbrakefull_impl() ++ (mode := self._normalPI_impl() ++ mode := self._accelerate_impl())))))}"\n            },\n            {\n                "name": "initialize",\n                "args": [],\n                "body": "{(initializeMode; t := 0.0; x := 0.0)}"\n            },\n            {\n                "name": "controlFunc",\n                "args": [],\n                "body": "{((?((mode == self._ebrakeact_impl())); ?((((15.0 <= v) and (v <= 40.0)) and (t == 2.5))); mode := self._ebrakefull_impl()) ++ ((?((mode == self._ebrakeact_impl())); ?(((((15.0 <= v) and (v <= 16.0)) and (0.0 <= t)) and (t <= 2.5))); t := 0.0; mode := self._sbrakeact_impl()) ++ ((?((mode == self._ebrakeact_impl())); ?(((t < 2.5) and (v > 15.0))); mode := mode) ++ ((?((mode == self._ebrakefull_impl())); ?((((15.0 <= v) and (v <= 16.0)) and (t == 2.5))); t := 0.0; mode := self._sbrakeact_impl()) ++ ((?((mode == self._ebrakefull_impl())); ?((v > 15.0)); mode := mode) ++ ((?((mode == self._sbrakeact_impl())); ?(((((18.0 <= v) and (v <= 20.0)) and (0.0 <= t)) and (t <= 1.3))); t := 0.0; mode := self._ebrakeact_impl()) ++ ((?((mode == self._sbrakeact_impl())); ?((((5.0 <= v) and (v <= 20.0)) and (t == 1.3))); mode := self._sbrakefull_impl()) ++ ((?((mode == self._sbrakeact_impl())); ?(((((5.0 <= v) and (v <= 11.0)) and (0.0 <= t)) and (t <= 1.3))); x := 0.0; mode := self._normalPI_impl()) ++ ((?((mode == self._sbrakeact_impl())); ?(((t < 1.3) and (v > 5.0))); mode := mode) ++ ((?((mode == self._sbrakefull_impl())); ?((((18.0 <= v) and (v <= 20.0)) and (t == 1.3))); t := 0.0; mode := self._ebrakeact_impl()) ++ ((?((mode == self._sbrakefull_impl())); ?((((5.0 <= v) and (v <= 11.0)) and (t == 1.3))); x := 0.0; mode := self._normalPI_impl()) ++ ((?((mode == self._sbrakefull_impl())); ?((v > 5.0)); mode := mode) ++ ((?((mode == self._normalPI_impl())); ?(((((13.0 <= v) and (v <= 15.0)) and (-(500.0) <= x)) and (x <= 500.0))); t := 0.0; mode := self._sbrakeact_impl()) ++ ((?((mode == self._normalPI_impl())); ?(((((-(15.0) <= v) and (v <= -(14.0))) and (-(500.0) <= x)) and (x <= 500.0))); mode := self._accelerate_impl()) ++ ((?((mode == self._normalPI_impl())); ?(((v > -(15.0)) and (v < 15.0))); mode := mode) ++ ((?((mode == self._accelerate_impl())); ?(((((-(6.0) <= v) and (v <= -(5.0))) and (-(500.0) <= x)) and (x <= 500.0))); x := 0.0; mode := self._normalPI_impl()) ++ (?((mode == self._accelerate_impl())); ?((v < -(5.0))); mode := mode)))))))))))))))))}"\n            },\n            {\n                "name": "continuousDynamics",\n                "args": [],\n                "body": "{((?((mode == self._ebrakeact_impl())); {(v\' = (-(t) - 2.5), x\' = 0.0, t\' = 1.0@@@DOMAIN_CONSTRAINT@@@sympy.And(sympy.And(sympy.And(sympy.Le(15.0, v), sympy.Le(v, 40.0)), sympy.Le(0.0, t)), sympy.Le(t, 2.5)))}) ++ ((?((mode == self._ebrakefull_impl())); {(v\' = -(5.0), x\' = 0.0, t\' = 0.0@@@DOMAIN_CONSTRAINT@@@sympy.And(sympy.And(sympy.Le(15.0, v), sympy.Le(v, 40.0)), sympy.Eq(t, 2.5)))}) ++ ((?((mode == self._sbrakeact_impl())); {(v\' = (-(t) - 1.2), x\' = 0.0, t\' = 0.5@@@DOMAIN_CONSTRAINT@@@sympy.And(sympy.And(sympy.And(sympy.Le(5.0, v), sympy.Le(v, 20.0)), sympy.Le(0.0, t)), sympy.Le(t, 1.3)))}) ++ ((?((mode == self._sbrakefull_impl())); {(v\' = -(2.5), x\' = 0.0, t\' = 0.0@@@DOMAIN_CONSTRAINT@@@sympy.And(sympy.And(sympy.Le(5.0, v), sympy.Le(v, 20.0)), sympy.Eq(t, 1.3)))}) ++ ((?((mode == self._normalPI_impl())); {(v\' = ((-(0.001) * x) - (0.052 * v)), x\' = v, t\' = 0.0@@@DOMAIN_CONSTRAINT@@@sympy.And(sympy.And(sympy.And(sympy.Le(-(15.0), v), sympy.Le(v, 15.0)), sympy.Le(-(500.0), x)), sympy.Le(x, 500.0)))}) ++ (?((mode == self._accelerate_impl())); {(v\' = 1.5, x\' = 0.0, t\' = 0.0@@@DOMAIN_CONSTRAINT@@@sympy.And(sympy.Le(-(20.0), v), sympy.Le(v, -(5.0))))}))))))}"\n            },\n            {\n                "name": "transitionAndFlow",\n                "args": [],\n                "body": "{(controlFunc; continuousDynamics)}"\n            },\n            {\n                "name": "cruiseLoop",\n                "args": [],\n                "body": "({transitionAndFlow})*"\n            },\n            {\n                "name": "cruiseControl",\n                "args": [],\n                "body": "{(initialize; cruiseLoop)}"\n            }\n        ],\n        "problem": {\n            "initial_condition": "True",\n            "program": "",\n            "goal": "True",\n            "loop_invariants": [],\n            "differential_invariants": [],\n            "differential_ghosts": []\n        },\n        "action_space_type": "discrete",\n        "action_space_bounds": {\n            "n": 1\n        },\n        "observation_space_type": "continuous",\n        "observation_space_bounds": {\n            "x": {\n                "min": "-500.0",\n                "max": "500.0"\n            },\n            "t": {\n                "min": "0.0",\n                "max": "2.5"\n            },\n            "v": {\n                "min": "-20.0",\n                "max": "40.0"\n            },\n            "mode": {\n                "min": "0.0",\n                "max": "5.0"\n            }\n        },\n        "reward_function_body": "((0.0) + ((0.0) if (self._cruiseSafe_impl()) else (-100))) + ((-(abs(v))) + ((10) if (self._nearZeroSpeed_impl()) else (0.0)))",\n        "termination_condition_body": "False",\n        "max_episode_steps": 200,\n        "time_step": 0.01,\n        "ode_solver_method": "DOP853",\n        "ode_solver_step_size": 0.001,\n        "ode_solver_tolerance": 1e-06,\n        "initial_state_sampling_config": {\n            "x": {\n                "min": "-500.0",\n                "max": "500.0"\n            },\n            "t": {\n                "min": "0.0",\n                "max": "2.5"\n            },\n            "v": {\n                "min": "-20.0",\n                "max": "40.0"\n            },\n            "mode": {\n                "min": "0.0",\n                "max": "5.0"\n            }\n        },\n        "seed": null,\n        "render_mode": null,\n        "metadata": {\n            "render_modes": [\n                "human",\n                "rgb_array"\n            ],\n            "render_fps": 30\n        },\n        "initial_state_buffer_config": {\n            "enabled": false,\n            "capacity": 100,\n            "populate_on_init": true,\n            "max_populate_attempts": 1000,\n            "strategy": "random"\n        },\n        "control_flow_program_ast_json": {\n            "_type": "Choice",\n            "left": {\n                "_type": "Sequence",\n                "programs": [\n                    {\n                        "_type": "Test",\n                        "condition": {\n                            "_type": "Comparison",\n                            "op": "=",\n                            "left": {\n                                "_type": "VarRef",\n                                "name": "mode"\n                            },\n                            "right": {\n                                "_type": "FunctionCall",\n                                "name": "ebrakeact",\n                                "args": []\n                            }\n                        }\n                    },\n                    {\n                        "_type": "Test",\n                        "condition": {\n                            "_type": "BinaryFormulaOp",\n                            "op": "&",\n                            "left": {\n                                "_type": "BinaryFormulaOp",\n                                "op": "&",\n                                "left": {\n                                    "_type": "Comparison",\n                                    "op": "<=",\n                                    "left": {\n                                        "_type": "Literal",\n                                        "value": 15.0\n                                    },\n                                    "right": {\n                                        "_type": "VarRef",\n                                        "name": "v"\n                                    }\n                                },\n                                "right": {\n                                    "_type": "Comparison",\n                                    "op": "<=",\n                                    "left": {\n                                        "_type": "VarRef",\n                                        "name": "v"\n                                    },\n                                    "right": {\n                                        "_type": "Literal",\n                                        "value": 40.0\n                                    }\n                                }\n                            },\n                            "right": {\n                                "_type": "Comparison",\n                                "op": "=",\n                                "left": {\n                                    "_type": "VarRef",\n                                    "name": "t"\n                                },\n                                "right": {\n                                    "_type": "Literal",\n                                    "value": 2.5\n                                }\n                            }\n                        }\n                    },\n                    {\n                        "_type": "Assignment",\n                        "var": {\n                            "_type": "VarRef",\n                            "name": "mode"\n                        },\n                        "expr": {\n                            "_type": "FunctionCall",\n                            "name": "ebrakefull",\n                            "args": []\n                        }\n                    }\n                ]\n            },\n            "right": {\n                "_type": "Choice",\n                "left": {\n                    "_type": "Sequence",\n                    "programs": [\n                        {\n                            "_type": "Test",\n                            "condition": {\n                                "_type": "Comparison",\n                                "op": "=",\n                                "left": {\n                                    "_type": "VarRef",\n                                    "name": "mode"\n                                },\n                                "right": {\n                                    "_type": "FunctionCall",\n                                    "name": "ebrakeact",\n                                    "args": []\n                                }\n                            }\n                        },\n                        {\n                            "_type": "Test",\n                            "condition": {\n                                "_type": "BinaryFormulaOp",\n                                "op": "&",\n                                "left": {\n                                    "_type": "BinaryFormulaOp",\n                                    "op": "&",\n                                    "left": {\n                                        "_type": "BinaryFormulaOp",\n                                        "op": "&",\n                                        "left": {\n                                            "_type": "Comparison",\n                                            "op": "<=",\n                                            "left": {\n                                                "_type": "Literal",\n                                                "value": 15.0\n                                            },\n                                            "right": {\n                                                "_type": "VarRef",\n                                                "name": "v"\n                                            }\n                                        },\n                                        "right": {\n                                            "_type": "Comparison",\n                                            "op": "<=",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "v"\n                                            },\n                                            "right": {\n                                                "_type": "Literal",\n                                                "value": 16.0\n                                            }\n                                        }\n                                    },\n                                    "right": {\n                                        "_type": "Comparison",\n                                        "op": "<=",\n                                        "left": {\n                                            "_type": "Literal",\n                                            "value": 0.0\n                                        },\n                                        "right": {\n                                            "_type": "VarRef",\n                                            "name": "t"\n                                        }\n                                    }\n                                },\n                                "right": {\n                                    "_type": "Comparison",\n                                    "op": "<=",\n                                    "left": {\n                                        "_type": "VarRef",\n                                        "name": "t"\n                                    },\n                                    "right": {\n                                        "_type": "Literal",\n                                        "value": 2.5\n                                    }\n                                }\n                            }\n                        },\n                        {\n                            "_type": "Assignment",\n                            "var": {\n                                "_type": "VarRef",\n                                "name": "t"\n                            },\n                            "expr": {\n                                "_type": "Literal",\n                                "value": 0.0\n                            }\n                        },\n                        {\n                            "_type": "Assignment",\n                            "var": {\n                                "_type": "VarRef",\n                                "name": "mode"\n                            },\n                            "expr": {\n                                "_type": "FunctionCall",\n                                "name": "sbrakeact",\n                                "args": []\n                            }\n                        }\n                    ]\n                },\n                "right": {\n                    "_type": "Choice",\n                    "left": {\n                        "_type": "Sequence",\n                        "programs": [\n                            {\n                                "_type": "Test",\n                                "condition": {\n                                    "_type": "Comparison",\n                                    "op": "=",\n                                    "left": {\n                                        "_type": "VarRef",\n                                        "name": "mode"\n                                    },\n                                    "right": {\n                                        "_type": "FunctionCall",\n                                        "name": "ebrakeact",\n                                        "args": []\n                                    }\n                                }\n                            },\n                            {\n                                "_type": "Test",\n                                "condition": {\n                                    "_type": "BinaryFormulaOp",\n                                    "op": "&",\n                                    "left": {\n                                        "_type": "Comparison",\n                                        "op": "<",\n                                        "left": {\n                                            "_type": "VarRef",\n                                            "name": "t"\n                                        },\n                                        "right": {\n                                            "_type": "Literal",\n                                            "value": 2.5\n                                        }\n                                    },\n                                    "right": {\n                                        "_type": "Comparison",\n                                        "op": ">",\n                                        "left": {\n                                            "_type": "VarRef",\n                                            "name": "v"\n                                        },\n                                        "right": {\n                                            "_type": "Literal",\n                                            "value": 15.0\n                                        }\n                                    }\n                                }\n                            },\n                            {\n                                "_type": "Assignment",\n                                "var": {\n                                    "_type": "VarRef",\n                                    "name": "mode"\n                                },\n                                "expr": {\n                                    "_type": "VarRef",\n                                    "name": "mode"\n                                }\n                            }\n                        ]\n                    },\n                    "right": {\n                        "_type": "Choice",\n                        "left": {\n                            "_type": "Sequence",\n                            "programs": [\n                                {\n                                    "_type": "Test",\n                                    "condition": {\n                                        "_type": "Comparison",\n                                        "op": "=",\n                                        "left": {\n                                            "_type": "VarRef",\n                                            "name": "mode"\n                                        },\n                                        "right": {\n                                            "_type": "FunctionCall",\n                                            "name": "ebrakefull",\n                                            "args": []\n                                        }\n                                    }\n                                },\n                                {\n                                    "_type": "Test",\n                                    "condition": {\n                                        "_type": "BinaryFormulaOp",\n                                        "op": "&",\n                                        "left": {\n                                            "_type": "BinaryFormulaOp",\n                                            "op": "&",\n                                            "left": {\n                                                "_type": "Comparison",\n                                                "op": "<=",\n                                                "left": {\n                                                    "_type": "Literal",\n                                                    "value": 15.0\n                                                },\n                                                "right": {\n                                                    "_type": "VarRef",\n                                                    "name": "v"\n                                                }\n                                            },\n                                            "right": {\n                                                "_type": "Comparison",\n                                                "op": "<=",\n                                                "left": {\n                                                    "_type": "VarRef",\n                                                    "name": "v"\n                                                },\n                                                "right": {\n                                                    "_type": "Literal",\n                                                    "value": 16.0\n                                                }\n                                            }\n                                        },\n                                        "right": {\n                                            "_type": "Comparison",\n                                            "op": "=",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "t"\n                                            },\n                                            "right": {\n                                                "_type": "Literal",\n                                                "value": 2.5\n                                            }\n                                        }\n                                    }\n                                },\n                                {\n                                    "_type": "Assignment",\n                                    "var": {\n                                        "_type": "VarRef",\n                                        "name": "t"\n                                    },\n                                    "expr": {\n                                        "_type": "Literal",\n                                        "value": 0.0\n                                    }\n                                },\n                                {\n                                    "_type": "Assignment",\n                                    "var": {\n                                        "_type": "VarRef",\n                                        "name": "mode"\n                                    },\n                                    "expr": {\n                                        "_type": "FunctionCall",\n                                        "name": "sbrakeact",\n                                        "args": []\n                                    }\n                                }\n                            ]\n                        },\n                        "right": {\n                            "_type": "Choice",\n                            "left": {\n                                "_type": "Sequence",\n                                "programs": [\n                                    {\n                                        "_type": "Test",\n                                        "condition": {\n                                            "_type": "Comparison",\n                                            "op": "=",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "mode"\n                                            },\n                                            "right": {\n                                                "_type": "FunctionCall",\n                                                "name": "ebrakefull",\n                                                "args": []\n                                            }\n                                        }\n                                    },\n                                    {\n                                        "_type": "Test",\n                                        "condition": {\n                                            "_type": "Comparison",\n                                            "op": ">",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "v"\n                                            },\n                                            "right": {\n                                                "_type": "Literal",\n                                                "value": 15.0\n                                            }\n                                        }\n                                    },\n                                    {\n                                        "_type": "Assignment",\n                                        "var": {\n                                            "_type": "VarRef",\n                                            "name": "mode"\n                                        },\n                                        "expr": {\n                                            "_type": "VarRef",\n                                            "name": "mode"\n                                        }\n                                    }\n                                ]\n                            },\n                            "right": {\n                                "_type": "Choice",\n                                "left": {\n                                    "_type": "Sequence",\n                                    "programs": [\n                                        {\n                                            "_type": "Test",\n                                            "condition": {\n                                                "_type": "Comparison",\n                                                "op": "=",\n                                                "left": {\n                                                    "_type": "VarRef",\n                                                    "name": "mode"\n                                                },\n                                                "right": {\n                                                    "_type": "FunctionCall",\n                                                    "name": "sbrakeact",\n                                                    "args": []\n                                                }\n                                            }\n                                        },\n                                        {\n                                            "_type": "Test",\n                                            "condition": {\n                                                "_type": "BinaryFormulaOp",\n                                                "op": "&",\n                                                "left": {\n                                                    "_type": "BinaryFormulaOp",\n                                                    "op": "&",\n                                                    "left": {\n                                                        "_type": "BinaryFormulaOp",\n                                                        "op": "&",\n                                                        "left": {\n                                                            "_type": "Comparison",\n                                                            "op": "<=",\n                                                            "left": {\n                                                                "_type": "Literal",\n                                                                "value": 18.0\n                                                            },\n                                                            "right": {\n                                                                "_type": "VarRef",\n                                                                "name": "v"\n                                                            }\n                                                        },\n                                                        "right": {\n                                                            "_type": "Comparison",\n                                                            "op": "<=",\n                                                            "left": {\n                                                                "_type": "VarRef",\n                                                                "name": "v"\n                                                            },\n                                                            "right": {\n                                                                "_type": "Literal",\n                                                                "value": 20.0\n                                                            }\n                                                        }\n                                                    },\n                                                    "right": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "Literal",\n                                                            "value": 0.0\n                                                        },\n                                                        "right": {\n                                                            "_type": "VarRef",\n                                                            "name": "t"\n                                                        }\n                                                    }\n                                                },\n                                                "right": {\n                                                    "_type": "Comparison",\n                                                    "op": "<=",\n                                                    "left": {\n                                                        "_type": "VarRef",\n                                                        "name": "t"\n                                                    },\n                                                    "right": {\n                                                        "_type": "Literal",\n                                                        "value": 1.3\n                                                    }\n                                                }\n                                            }\n                                        },\n                                        {\n                                            "_type": "Assignment",\n                                            "var": {\n                                                "_type": "VarRef",\n                                                "name": "t"\n                                            },\n                                            "expr": {\n                                                "_type": "Literal",\n                                                "value": 0.0\n                                            }\n                                        },\n                                        {\n                                            "_type": "Assignment",\n                                            "var": {\n                                                "_type": "VarRef",\n                                                "name": "mode"\n                                            },\n                                            "expr": {\n                                                "_type": "FunctionCall",\n                                                "name": "ebrakeact",\n                                                "args": []\n                                            }\n                                        }\n                                    ]\n                                },\n                                "right": {\n                                    "_type": "Choice",\n                                    "left": {\n                                        "_type": "Sequence",\n                                        "programs": [\n                                            {\n                                                "_type": "Test",\n                                                "condition": {\n                                                    "_type": "Comparison",\n                                                    "op": "=",\n                                                    "left": {\n                                                        "_type": "VarRef",\n                                                        "name": "mode"\n                                                    },\n                                                    "right": {\n                                                        "_type": "FunctionCall",\n                                                        "name": "sbrakeact",\n                                                        "args": []\n                                                    }\n                                                }\n                                            },\n                                            {\n                                                "_type": "Test",\n                                                "condition": {\n                                                    "_type": "BinaryFormulaOp",\n                                                    "op": "&",\n                                                    "left": {\n                                                        "_type": "BinaryFormulaOp",\n                                                        "op": "&",\n                                                        "left": {\n                                                            "_type": "Comparison",\n                                                            "op": "<=",\n                                                            "left": {\n                                                                "_type": "Literal",\n                                                                "value": 5.0\n                                                            },\n                                                            "right": {\n                                                                "_type": "VarRef",\n                                                                "name": "v"\n                                                            }\n                                                        },\n                                                        "right": {\n                                                            "_type": "Comparison",\n                                                            "op": "<=",\n                                                            "left": {\n                                                                "_type": "VarRef",\n                                                                "name": "v"\n                                                            },\n                                                            "right": {\n                                                                "_type": "Literal",\n                                                                "value": 20.0\n                                                            }\n                                                        }\n                                                    },\n                                                    "right": {\n                                                        "_type": "Comparison",\n                                                        "op": "=",\n                                                        "left": {\n                                                            "_type": "VarRef",\n                                                            "name": "t"\n                                                        },\n                                                        "right": {\n                                                            "_type": "Literal",\n                                                            "value": 1.3\n                                                        }\n                                                    }\n                                                }\n                                            },\n                                            {\n                                                "_type": "Assignment",\n                                                "var": {\n                                                    "_type": "VarRef",\n                                                    "name": "mode"\n                                                },\n                                                "expr": {\n                                                    "_type": "FunctionCall",\n                                                    "name": "sbrakefull",\n                                                    "args": []\n                                                }\n                                            }\n                                        ]\n                                    },\n                                    "right": {\n                                        "_type": "Choice",\n                                        "left": {\n                                            "_type": "Sequence",\n                                            "programs": [\n                                                {\n                                                    "_type": "Test",\n                                                    "condition": {\n                                                        "_type": "Comparison",\n                                                        "op": "=",\n                                                        "left": {\n                                                            "_type": "VarRef",\n                                                            "name": "mode"\n                                                        },\n                                                        "right": {\n                                                            "_type": "FunctionCall",\n                                                            "name": "sbrakeact",\n                                                            "args": []\n                                                        }\n                                                    }\n                                                },\n                                                {\n                                                    "_type": "Test",\n                                                    "condition": {\n                                                        "_type": "BinaryFormulaOp",\n                                                        "op": "&",\n                                                        "left": {\n                                                            "_type": "BinaryFormulaOp",\n                                                            "op": "&",\n                                                            "left": {\n                                                                "_type": "BinaryFormulaOp",\n                                                                "op": "&",\n                                                                "left": {\n                                                                    "_type": "Comparison",\n                                                                    "op": "<=",\n                                                                    "left": {\n                                                                        "_type": "Literal",\n                                                                        "value": 5.0\n                                                                    },\n                                                                    "right": {\n                                                                        "_type": "VarRef",\n                                                                        "name": "v"\n                                                                    }\n                                                                },\n                                                                "right": {\n                                                                    "_type": "Comparison",\n                                                                    "op": "<=",\n                                                                    "left": {\n                                                                        "_type": "VarRef",\n                                                                        "name": "v"\n                                                                    },\n                                                                    "right": {\n                                                                        "_type": "Literal",\n                                                                        "value": 11.0\n                                                                    }\n                                                                }\n                                                            },\n                                                            "right": {\n                                                                "_type": "Comparison",\n                                                                "op": "<=",\n                                                                "left": {\n                                                                    "_type": "Literal",\n                                                                    "value": 0.0\n                                                                },\n                                                                "right": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "t"\n                                                                }\n                                                            }\n                                                        },\n                                                        "right": {\n                                                            "_type": "Comparison",\n                                                            "op": "<=",\n                                                            "left": {\n                                                                "_type": "VarRef",\n                                                                "name": "t"\n                                                            },\n                                                            "right": {\n                                                                "_type": "Literal",\n                                                                "value": 1.3\n                                                            }\n                                                        }\n                                                    }\n                                                },\n                                                {\n                                                    "_type": "Assignment",\n                                                    "var": {\n                                                        "_type": "VarRef",\n                                                        "name": "x"\n                                                    },\n                                                    "expr": {\n                                                        "_type": "Literal",\n                                                        "value": 0.0\n                                                    }\n                                                },\n                                                {\n                                                    "_type": "Assignment",\n                                                    "var": {\n                                                        "_type": "VarRef",\n                                                        "name": "mode"\n                                                    },\n                                                    "expr": {\n                                                        "_type": "FunctionCall",\n                                                        "name": "normalPI",\n                                                        "args": []\n                                                    }\n                                                }\n                                            ]\n                                        },\n                                        "right": {\n                                            "_type": "Choice",\n                                            "left": {\n                                                "_type": "Sequence",\n                                                "programs": [\n                                                    {\n                                                        "_type": "Test",\n                                                        "condition": {\n                                                            "_type": "Comparison",\n                                                            "op": "=",\n                                                            "left": {\n                                                                "_type": "VarRef",\n                                                                "name": "mode"\n                                                            },\n                                                            "right": {\n                                                                "_type": "FunctionCall",\n                                                                "name": "sbrakeact",\n                                                                "args": []\n                                                            }\n                                                        }\n                                                    },\n                                                    {\n                                                        "_type": "Test",\n                                                        "condition": {\n                                                            "_type": "BinaryFormulaOp",\n                                                            "op": "&",\n                                                            "left": {\n                                                                "_type": "Comparison",\n                                                                "op": "<",\n                                                                "left": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "t"\n                                                                },\n                                                                "right": {\n                                                                    "_type": "Literal",\n                                                                    "value": 1.3\n                                                                }\n                                                            },\n                                                            "right": {\n                                                                "_type": "Comparison",\n                                                                "op": ">",\n                                                                "left": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "v"\n                                                                },\n                                                                "right": {\n                                                                    "_type": "Literal",\n                                                                    "value": 5.0\n                                                                }\n                                                            }\n                                                        }\n                                                    },\n                                                    {\n                                                        "_type": "Assignment",\n                                                        "var": {\n                                                            "_type": "VarRef",\n                                                            "name": "mode"\n                                                        },\n                                                        "expr": {\n                                                            "_type": "VarRef",\n                                                            "name": "mode"\n                                                        }\n                                                    }\n                                                ]\n                                            },\n                                            "right": {\n                                                "_type": "Choice",\n                                                "left": {\n                                                    "_type": "Sequence",\n                                                    "programs": [\n                                                        {\n                                                            "_type": "Test",\n                                                            "condition": {\n                                                                "_type": "Comparison",\n                                                                "op": "=",\n                                                                "left": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "mode"\n                                                                },\n                                                                "right": {\n                                                                    "_type": "FunctionCall",\n                                                                    "name": "sbrakefull",\n                                                                    "args": []\n                                                                }\n                                                            }\n                                                        },\n                                                        {\n                                                            "_type": "Test",\n                                                            "condition": {\n                                                                "_type": "BinaryFormulaOp",\n                                                                "op": "&",\n                                                                "left": {\n                                                                    "_type": "BinaryFormulaOp",\n                                                                    "op": "&",\n                                                                    "left": {\n                                                                        "_type": "Comparison",\n                                                                        "op": "<=",\n                                                                        "left": {\n                                                                            "_type": "Literal",\n                                                                            "value": 18.0\n                                                                        },\n                                                                        "right": {\n                                                                            "_type": "VarRef",\n                                                                            "name": "v"\n                                                                        }\n                                                                    },\n                                                                    "right": {\n                                                                        "_type": "Comparison",\n                                                                        "op": "<=",\n                                                                        "left": {\n                                                                            "_type": "VarRef",\n                                                                            "name": "v"\n                                                                        },\n                                                                        "right": {\n                                                                            "_type": "Literal",\n                                                                            "value": 20.0\n                                                                        }\n                                                                    }\n                                                                },\n                                                                "right": {\n                                                                    "_type": "Comparison",\n                                                                    "op": "=",\n                                                                    "left": {\n                                                                        "_type": "VarRef",\n                                                                        "name": "t"\n                                                                    },\n                                                                    "right": {\n                                                                        "_type": "Literal",\n                                                                        "value": 1.3\n                                                                    }\n                                                                }\n                                                            }\n                                                        },\n                                                        {\n                                                            "_type": "Assignment",\n                                                            "var": {\n                                                                "_type": "VarRef",\n                                                                "name": "t"\n                                                            },\n                                                            "expr": {\n                                                                "_type": "Literal",\n                                                                "value": 0.0\n                                                            }\n                                                        },\n                                                        {\n                                                            "_type": "Assignment",\n                                                            "var": {\n                                                                "_type": "VarRef",\n                                                                "name": "mode"\n                                                            },\n                                                            "expr": {\n                                                                "_type": "FunctionCall",\n                                                                "name": "ebrakeact",\n                                                                "args": []\n                                                            }\n                                                        }\n                                                    ]\n                                                },\n                                                "right": {\n                                                    "_type": "Choice",\n                                                    "left": {\n                                                        "_type": "Sequence",\n                                                        "programs": [\n                                                            {\n                                                                "_type": "Test",\n                                                                "condition": {\n                                                                    "_type": "Comparison",\n                                                                    "op": "=",\n                                                                    "left": {\n                                                                        "_type": "VarRef",\n                                                                        "name": "mode"\n                                                                    },\n                                                                    "right": {\n                                                                        "_type": "FunctionCall",\n                                                                        "name": "sbrakefull",\n                                                                        "args": []\n                                                                    }\n                                                                }\n                                                            },\n                                                            {\n                                                                "_type": "Test",\n                                                                "condition": {\n                                                                    "_type": "BinaryFormulaOp",\n                                                                    "op": "&",\n                                                                    "left": {\n                                                                        "_type": "BinaryFormulaOp",\n                                                                        "op": "&",\n                                                                        "left": {\n                                                                            "_type": "Comparison",\n                                                                            "op": "<=",\n                                                                            "left": {\n                                                                                "_type": "Literal",\n                                                                                "value": 5.0\n                                                                            },\n                                                                            "right": {\n                                                                                "_type": "VarRef",\n                                                                                "name": "v"\n                                                                            }\n                                                                        },\n                                                                        "right": {\n                                                                            "_type": "Comparison",\n                                                                            "op": "<=",\n                                                                            "left": {\n                                                                                "_type": "VarRef",\n                                                                                "name": "v"\n                                                                            },\n                                                                            "right": {\n                                                                                "_type": "Literal",\n                                                                                "value": 11.0\n                                                                            }\n                                                                        }\n                                                                    },\n                                                                    "right": {\n                                                                        "_type": "Comparison",\n                                                                        "op": "=",\n                                                                        "left": {\n                                                                            "_type": "VarRef",\n                                                                            "name": "t"\n                                                                        },\n                                                                        "right": {\n                                                                            "_type": "Literal",\n                                                                            "value": 1.3\n                                                                        }\n                                                                    }\n                                                                }\n                                                            },\n                                                            {\n                                                                "_type": "Assignment",\n                                                                "var": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "x"\n                                                                },\n                                                                "expr": {\n                                                                    "_type": "Literal",\n                                                                    "value": 0.0\n                                                                }\n                                                            },\n                                                            {\n                                                                "_type": "Assignment",\n                                                                "var": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "mode"\n                                                                },\n                                                                "expr": {\n                                                                    "_type": "FunctionCall",\n                                                                    "name": "normalPI",\n                                                                    "args": []\n                                                                }\n                                                            }\n                                                        ]\n                                                    },\n                                                    "right": {\n                                                        "_type": "Choice",\n                                                        "left": {\n                                                            "_type": "Sequence",\n                                                            "programs": [\n                                                                {\n                                                                    "_type": "Test",\n                                                                    "condition": {\n                                                                        "_type": "Comparison",\n                                                                        "op": "=",\n                                                                        "left": {\n                                                                            "_type": "VarRef",\n                                                                            "name": "mode"\n                                                                        },\n                                                                        "right": {\n                                                                            "_type": "FunctionCall",\n                                                                            "name": "sbrakefull",\n                                                                            "args": []\n                                                                        }\n                                                                    }\n                                                                },\n                                                                {\n                                                                    "_type": "Test",\n                                                                    "condition": {\n                                                                        "_type": "Comparison",\n                                                                        "op": ">",\n                                                                        "left": {\n                                                                            "_type": "VarRef",\n                                                                            "name": "v"\n                                                                        },\n                                                                        "right": {\n                                                                            "_type": "Literal",\n                                                                            "value": 5.0\n                                                                        }\n                                                                    }\n                                                                },\n                                                                {\n                                                                    "_type": "Assignment",\n                                                                    "var": {\n                                                                        "_type": "VarRef",\n                                                                        "name": "mode"\n                                                                    },\n                                                                    "expr": {\n                                                                        "_type": "VarRef",\n                                                                        "name": "mode"\n                                                                    }\n                                                                }\n                                                            ]\n                                                        },\n                                                        "right": {\n                                                            "_type": "Choice",\n                                                            "left": {\n                                                                "_type": "Sequence",\n                                                                "programs": [\n                                                                    {\n                                                                        "_type": "Test",\n                                                                        "condition": {\n                                                                            "_type": "Comparison",\n                                                                            "op": "=",\n                                                                            "left": {\n                                                                                "_type": "VarRef",\n                                                                                "name": "mode"\n                                                                            },\n                                                                            "right": {\n                                                                                "_type": "FunctionCall",\n                                                                                "name": "normalPI",\n                                                                                "args": []\n                                                                            }\n                                                                        }\n                                                                    },\n                                                                    {\n                                                                        "_type": "Test",\n                                                                        "condition": {\n                                                                            "_type": "BinaryFormulaOp",\n                                                                            "op": "&",\n                                                                            "left": {\n                                                                                "_type": "BinaryFormulaOp",\n                                                                                "op": "&",\n                                                                                "left": {\n                                                                                    "_type": "BinaryFormulaOp",\n                                                                                    "op": "&",\n                                                                                    "left": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": "<=",\n                                                                                        "left": {\n                                                                                            "_type": "Literal",\n                                                                                            "value": 13.0\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "v"\n                                                                                        }\n                                                                                    },\n                                                                                    "right": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": "<=",\n                                                                                        "left": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "v"\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "Literal",\n                                                                                            "value": 15.0\n                                                                                        }\n                                                                                    }\n                                                                                },\n                                                                                "right": {\n                                                                                    "_type": "Comparison",\n                                                                                    "op": "<=",\n                                                                                    "left": {\n                                                                                        "_type": "UnaryTermOp",\n                                                                                        "op": "-",\n                                                                                        "operand": {\n                                                                                            "_type": "Literal",\n                                                                                            "value": 500.0\n                                                                                        }\n                                                                                    },\n                                                                                    "right": {\n                                                                                        "_type": "VarRef",\n                                                                                        "name": "x"\n                                                                                    }\n                                                                                }\n                                                                            },\n                                                                            "right": {\n                                                                                "_type": "Comparison",\n                                                                                "op": "<=",\n                                                                                "left": {\n                                                                                    "_type": "VarRef",\n                                                                                    "name": "x"\n                                                                                },\n                                                                                "right": {\n                                                                                    "_type": "Literal",\n                                                                                    "value": 500.0\n                                                                                }\n                                                                            }\n                                                                        }\n                                                                    },\n                                                                    {\n                                                                        "_type": "Assignment",\n                                                                        "var": {\n                                                                            "_type": "VarRef",\n                                                                            "name": "t"\n                                                                        },\n                                                                        "expr": {\n                                                                            "_type": "Literal",\n                                                                            "value": 0.0\n                                                                        }\n                                                                    },\n                                                                    {\n                                                                        "_type": "Assignment",\n                                                                        "var": {\n                                                                            "_type": "VarRef",\n                                                                            "name": "mode"\n                                                                        },\n                                                                        "expr": {\n                                                                            "_type": "FunctionCall",\n                                                                            "name": "sbrakeact",\n                                                                            "args": []\n                                                                        }\n                                                                    }\n                                                                ]\n                                                            },\n                                                            "right": {\n                                                                "_type": "Choice",\n                                                                "left": {\n                                                                    "_type": "Sequence",\n                                                                    "programs": [\n                                                                        {\n                                                                            "_type": "Test",\n                                                                            "condition": {\n                                                                                "_type": "Comparison",\n                                                                                "op": "=",\n                                                                                "left": {\n                                                                                    "_type": "VarRef",\n                                                                                    "name": "mode"\n                                                                                },\n                                                                                "right": {\n                                                                                    "_type": "FunctionCall",\n                                                                                    "name": "normalPI",\n                                                                                    "args": []\n                                                                                }\n                                                                            }\n                                                                        },\n                                                                        {\n                                                                            "_type": "Test",\n                                                                            "condition": {\n                                                                                "_type": "BinaryFormulaOp",\n                                                                                "op": "&",\n                                                                                "left": {\n                                                                                    "_type": "BinaryFormulaOp",\n                                                                                    "op": "&",\n                                                                                    "left": {\n                                                                                        "_type": "BinaryFormulaOp",\n                                                                                        "op": "&",\n                                                                                        "left": {\n                                                                                            "_type": "Comparison",\n                                                                                            "op": "<=",\n                                                                                            "left": {\n                                                                                                "_type": "UnaryTermOp",\n                                                                                                "op": "-",\n                                                                                                "operand": {\n                                                                                                    "_type": "Literal",\n                                                                                                    "value": 15.0\n                                                                                                }\n                                                                                            },\n                                                                                            "right": {\n                                                                                                "_type": "VarRef",\n                                                                                                "name": "v"\n                                                                                            }\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "Comparison",\n                                                                                            "op": "<=",\n                                                                                            "left": {\n                                                                                                "_type": "VarRef",\n                                                                                                "name": "v"\n                                                                                            },\n                                                                                            "right": {\n                                                                                                "_type": "UnaryTermOp",\n                                                                                                "op": "-",\n                                                                                                "operand": {\n                                                                                                    "_type": "Literal",\n                                                                                                    "value": 14.0\n                                                                                                }\n                                                                                            }\n                                                                                        }\n                                                                                    },\n                                                                                    "right": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": "<=",\n                                                                                        "left": {\n                                                                                            "_type": "UnaryTermOp",\n                                                                                            "op": "-",\n                                                                                            "operand": {\n                                                                                                "_type": "Literal",\n                                                                                                "value": 500.0\n                                                                                            }\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "x"\n                                                                                        }\n                                                                                    }\n                                                                                },\n                                                                                "right": {\n                                                                                    "_type": "Comparison",\n                                                                                    "op": "<=",\n                                                                                    "left": {\n                                                                                        "_type": "VarRef",\n                                                                                        "name": "x"\n                                                                                    },\n                                                                                    "right": {\n                                                                                        "_type": "Literal",\n                                                                                        "value": 500.0\n                                                                                    }\n                                                                                }\n                                                                            }\n                                                                        },\n                                                                        {\n                                                                            "_type": "Assignment",\n                                                                            "var": {\n                                                                                "_type": "VarRef",\n                                                                                "name": "mode"\n                                                                            },\n                                                                            "expr": {\n                                                                                "_type": "FunctionCall",\n                                                                                "name": "accelerate",\n                                                                                "args": []\n                                                                            }\n                                                                        }\n                                                                    ]\n                                                                },\n                                                                "right": {\n                                                                    "_type": "Choice",\n                                                                    "left": {\n                                                                        "_type": "Sequence",\n                                                                        "programs": [\n                                                                            {\n                                                                                "_type": "Test",\n                                                                                "condition": {\n                                                                                    "_type": "Comparison",\n                                                                                    "op": "=",\n                                                                                    "left": {\n                                                                                        "_type": "VarRef",\n                                                                                        "name": "mode"\n                                                                                    },\n                                                                                    "right": {\n                                                                                        "_type": "FunctionCall",\n                                                                                        "name": "normalPI",\n                                                                                        "args": []\n                                                                                    }\n                                                                                }\n                                                                            },\n                                                                            {\n                                                                                "_type": "Test",\n                                                                                "condition": {\n                                                                                    "_type": "BinaryFormulaOp",\n                                                                                    "op": "&",\n                                                                                    "left": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": ">",\n                                                                                        "left": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "v"\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "UnaryTermOp",\n                                                                                            "op": "-",\n                                                                                            "operand": {\n                                                                                                "_type": "Literal",\n                                                                                                "value": 15.0\n                                                                                            }\n                                                                                        }\n                                                                                    },\n                                                                                    "right": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": "<",\n                                                                                        "left": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "v"\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "Literal",\n                                                                                            "value": 15.0\n                                                                                        }\n                                                                                    }\n                                                                                }\n                                                                            },\n                                                                            {\n                                                                                "_type": "Assignment",\n                                                                                "var": {\n                                                                                    "_type": "VarRef",\n                                                                                    "name": "mode"\n                                                                                },\n                                                                                "expr": {\n                                                                                    "_type": "VarRef",\n                                                                                    "name": "mode"\n                                                                                }\n                                                                            }\n                                                                        ]\n                                                                    },\n                                                                    "right": {\n                                                                        "_type": "Choice",\n                                                                        "left": {\n                                                                            "_type": "Sequence",\n                                                                            "programs": [\n                                                                                {\n                                                                                    "_type": "Test",\n                                                                                    "condition": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": "=",\n                                                                                        "left": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "mode"\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "FunctionCall",\n                                                                                            "name": "accelerate",\n                                                                                            "args": []\n                                                                                        }\n                                                                                    }\n                                                                                },\n                                                                                {\n                                                                                    "_type": "Test",\n                                                                                    "condition": {\n                                                                                        "_type": "BinaryFormulaOp",\n                                                                                        "op": "&",\n                                                                                        "left": {\n                                                                                            "_type": "BinaryFormulaOp",\n                                                                                            "op": "&",\n                                                                                            "left": {\n                                                                                                "_type": "BinaryFormulaOp",\n                                                                                                "op": "&",\n                                                                                                "left": {\n                                                                                                    "_type": "Comparison",\n                                                                                                    "op": "<=",\n                                                                                                    "left": {\n                                                                                                        "_type": "UnaryTermOp",\n                                                                                                        "op": "-",\n                                                                                                        "operand": {\n                                                                                                            "_type": "Literal",\n                                                                                                            "value": 6.0\n                                                                                                        }\n                                                                                                    },\n                                                                                                    "right": {\n                                                                                                        "_type": "VarRef",\n                                                                                                        "name": "v"\n                                                                                                    }\n                                                                                                },\n                                                                                                "right": {\n                                                                                                    "_type": "Comparison",\n                                                                                                    "op": "<=",\n                                                                                                    "left": {\n                                                                                                        "_type": "VarRef",\n                                                                                                        "name": "v"\n                                                                                                    },\n                                                                                                    "right": {\n                                                                                                        "_type": "UnaryTermOp",\n                                                                                                        "op": "-",\n                                                                                                        "operand": {\n                                                                                                            "_type": "Literal",\n                                                                                                            "value": 5.0\n                                                                                                        }\n                                                                                                    }\n                                                                                                }\n                                                                                            },\n                                                                                            "right": {\n                                                                                                "_type": "Comparison",\n                                                                                                "op": "<=",\n                                                                                                "left": {\n                                                                                                    "_type": "UnaryTermOp",\n                                                                                                    "op": "-",\n                                                                                                    "operand": {\n                                                                                                        "_type": "Literal",\n                                                                                                        "value": 500.0\n                                                                                                    }\n                                                                                                },\n                                                                                                "right": {\n                                                                                                    "_type": "VarRef",\n                                                                                                    "name": "x"\n                                                                                                }\n                                                                                            }\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "Comparison",\n                                                                                            "op": "<=",\n                                                                                            "left": {\n                                                                                                "_type": "VarRef",\n                                                                                                "name": "x"\n                                                                                            },\n                                                                                            "right": {\n                                                                                                "_type": "Literal",\n                                                                                                "value": 500.0\n                                                                                            }\n                                                                                        }\n                                                                                    }\n                                                                                },\n                                                                                {\n                                                                                    "_type": "Assignment",\n                                                                                    "var": {\n                                                                                        "_type": "VarRef",\n                                                                                        "name": "x"\n                                                                                    },\n                                                                                    "expr": {\n                                                                                        "_type": "Literal",\n                                                                                        "value": 0.0\n                                                                                    }\n                                                                                },\n                                                                                {\n                                                                                    "_type": "Assignment",\n                                                                                    "var": {\n                                                                                        "_type": "VarRef",\n                                                                                        "name": "mode"\n                                                                                    },\n                                                                                    "expr": {\n                                                                                        "_type": "FunctionCall",\n                                                                                        "name": "normalPI",\n                                                                                        "args": []\n                                                                                    }\n                                                                                }\n                                                                            ]\n                                                                        },\n                                                                        "right": {\n                                                                            "_type": "Sequence",\n                                                                            "programs": [\n                                                                                {\n                                                                                    "_type": "Test",\n                                                                                    "condition": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": "=",\n                                                                                        "left": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "mode"\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "FunctionCall",\n                                                                                            "name": "accelerate",\n                                                                                            "args": []\n                                                                                        }\n                                                                                    }\n                                                                                },\n                                                                                {\n                                                                                    "_type": "Test",\n                                                                                    "condition": {\n                                                                                        "_type": "Comparison",\n                                                                                        "op": "<",\n                                                                                        "left": {\n                                                                                            "_type": "VarRef",\n                                                                                            "name": "v"\n                                                                                        },\n                                                                                        "right": {\n                                                                                            "_type": "UnaryTermOp",\n                                                                                            "op": "-",\n                                                                                            "operand": {\n                                                                                                "_type": "Literal",\n                                                                                                "value": 5.0\n                                                                                            }\n                                                                                        }\n                                                                                    }\n                                                                                },\n                                                                                {\n                                                                                    "_type": "Assignment",\n                                                                                    "var": {\n                                                                                        "_type": "VarRef",\n                                                                                        "name": "mode"\n                                                                                    },\n                                                                                    "expr": {\n                                                                                        "_type": "VarRef",\n                                                                                        "name": "mode"\n                                                                                    }\n                                                                                }\n                                                                            ]\n                                                                        }\n                                                                    }\n                                                                }\n                                                            }\n                                                        }\n                                                    }\n                                                }\n                                            }\n                                        }\n                                    }\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n        },\n        "simulation_mode": true,\n        "simulation_initialization_ast_json": {\n            "_type": "Sequence",\n            "programs": [\n                {\n                    "_type": "Choice",\n                    "left": {\n                        "_type": "Assignment",\n                        "var": {\n                            "_type": "VarRef",\n                            "name": "mode"\n                        },\n                        "expr": {\n                            "_type": "FunctionCall",\n                            "name": "ebrakeact",\n                            "args": []\n                        }\n                    },\n                    "right": {\n                        "_type": "Choice",\n                        "left": {\n                            "_type": "Assignment",\n                            "var": {\n                                "_type": "VarRef",\n                                "name": "mode"\n                            },\n                            "expr": {\n                                "_type": "FunctionCall",\n                                "name": "ebrakefull",\n                                "args": []\n                            }\n                        },\n                        "right": {\n                            "_type": "Choice",\n                            "left": {\n                                "_type": "Assignment",\n                                "var": {\n                                    "_type": "VarRef",\n                                    "name": "mode"\n                                },\n                                "expr": {\n                                    "_type": "FunctionCall",\n                                    "name": "sbrakeact",\n                                    "args": []\n                                }\n                            },\n                            "right": {\n                                "_type": "Choice",\n                                "left": {\n                                    "_type": "Assignment",\n                                    "var": {\n                                        "_type": "VarRef",\n                                        "name": "mode"\n                                    },\n                                    "expr": {\n                                        "_type": "FunctionCall",\n                                        "name": "sbrakefull",\n                                        "args": []\n                                    }\n                                },\n                                "right": {\n                                    "_type": "Choice",\n                                    "left": {\n                                        "_type": "Assignment",\n                                        "var": {\n                                            "_type": "VarRef",\n                                            "name": "mode"\n                                        },\n                                        "expr": {\n                                            "_type": "FunctionCall",\n                                            "name": "normalPI",\n                                            "args": []\n                                        }\n                                    },\n                                    "right": {\n                                        "_type": "Assignment",\n                                        "var": {\n                                            "_type": "VarRef",\n                                            "name": "mode"\n                                        },\n                                        "expr": {\n                                            "_type": "FunctionCall",\n                                            "name": "accelerate",\n                                            "args": []\n                                        }\n                                    }\n                                }\n                            }\n                        }\n                    }\n                },\n                {\n                    "_type": "Assignment",\n                    "var": {\n                        "_type": "VarRef",\n                        "name": "t"\n                    },\n                    "expr": {\n                        "_type": "Literal",\n                        "value": 0.0\n                    }\n                },\n                {\n                    "_type": "Assignment",\n                    "var": {\n                        "_type": "VarRef",\n                        "name": "x"\n                    },\n                    "expr": {\n                        "_type": "Literal",\n                        "value": 0.0\n                    }\n                }\n            ]\n        },\n        "simulation_step_ast_json": {\n            "_type": "Sequence",\n            "programs": [\n                {\n                    "_type": "Sequence",\n                    "programs": []\n                },\n                {\n                    "_type": "Choice",\n                    "left": {\n                        "_type": "Sequence",\n                        "programs": [\n                            {\n                                "_type": "Test",\n                                "condition": {\n                                    "_type": "Comparison",\n                                    "op": "=",\n                                    "left": {\n                                        "_type": "VarRef",\n                                        "name": "mode"\n                                    },\n                                    "right": {\n                                        "_type": "FunctionCall",\n                                        "name": "ebrakeact",\n                                        "args": []\n                                    }\n                                }\n                            },\n                            {\n                                "_type": "ODE",\n                                "equations": [\n                                    {\n                                        "_type": "ODEEquation",\n                                        "var": {\n                                            "_type": "VarRef",\n                                            "name": "v"\n                                        },\n                                        "expr": {\n                                            "_type": "BinaryTermOp",\n                                            "op": "-",\n                                            "left": {\n                                                "_type": "UnaryTermOp",\n                                                "op": "-",\n                                                "operand": {\n                                                    "_type": "VarRef",\n                                                    "name": "t"\n                                                }\n                                            },\n                                            "right": {\n                                                "_type": "Literal",\n                                                "value": 2.5\n                                            }\n                                        }\n                                    },\n                                    {\n                                        "_type": "ODEEquation",\n                                        "var": {\n                                            "_type": "VarRef",\n                                            "name": "x"\n                                        },\n                                        "expr": {\n                                            "_type": "Literal",\n                                            "value": 0.0\n                                        }\n                                    },\n                                    {\n                                        "_type": "ODEEquation",\n                                        "var": {\n                                            "_type": "VarRef",\n                                            "name": "t"\n                                        },\n                                        "expr": {\n                                            "_type": "Literal",\n                                            "value": 1.0\n                                        }\n                                    }\n                                ],\n                                "domain_constraint": {\n                                    "_type": "BinaryFormulaOp",\n                                    "op": "&",\n                                    "left": {\n                                        "_type": "BinaryFormulaOp",\n                                        "op": "&",\n                                        "left": {\n                                            "_type": "BinaryFormulaOp",\n                                            "op": "&",\n                                            "left": {\n                                                "_type": "Comparison",\n                                                "op": "<=",\n                                                "left": {\n                                                    "_type": "Literal",\n                                                    "value": 15.0\n                                                },\n                                                "right": {\n                                                    "_type": "VarRef",\n                                                    "name": "v"\n                                                }\n                                            },\n                                            "right": {\n                                                "_type": "Comparison",\n                                                "op": "<=",\n                                                "left": {\n                                                    "_type": "VarRef",\n                                                    "name": "v"\n                                                },\n                                                "right": {\n                                                    "_type": "Literal",\n                                                    "value": 40.0\n                                                }\n                                            }\n                                        },\n                                        "right": {\n                                            "_type": "Comparison",\n                                            "op": "<=",\n                                            "left": {\n                                                "_type": "Literal",\n                                                "value": 0.0\n                                            },\n                                            "right": {\n                                                "_type": "VarRef",\n                                                "name": "t"\n                                            }\n                                        }\n                                    },\n                                    "right": {\n                                        "_type": "Comparison",\n                                        "op": "<=",\n                                        "left": {\n                                            "_type": "VarRef",\n                                            "name": "t"\n                                        },\n                                        "right": {\n                                            "_type": "Literal",\n                                            "value": 2.5\n                                        }\n                                    }\n                                }\n                            }\n                        ]\n                    },\n                    "right": {\n                        "_type": "Choice",\n                        "left": {\n                            "_type": "Sequence",\n                            "programs": [\n                                {\n                                    "_type": "Test",\n                                    "condition": {\n                                        "_type": "Comparison",\n                                        "op": "=",\n                                        "left": {\n                                            "_type": "VarRef",\n                                            "name": "mode"\n                                        },\n                                        "right": {\n                                            "_type": "FunctionCall",\n                                            "name": "ebrakefull",\n                                            "args": []\n                                        }\n                                    }\n                                },\n                                {\n                                    "_type": "ODE",\n                                    "equations": [\n                                        {\n                                            "_type": "ODEEquation",\n                                            "var": {\n                                                "_type": "VarRef",\n                                                "name": "v"\n                                            },\n                                            "expr": {\n                                                "_type": "UnaryTermOp",\n                                                "op": "-",\n                                                "operand": {\n                                                    "_type": "Literal",\n                                                    "value": 5.0\n                                                }\n                                            }\n                                        },\n                                        {\n                                            "_type": "ODEEquation",\n                                            "var": {\n                                                "_type": "VarRef",\n                                                "name": "x"\n                                            },\n                                            "expr": {\n                                                "_type": "Literal",\n                                                "value": 0.0\n                                            }\n                                        },\n                                        {\n                                            "_type": "ODEEquation",\n                                            "var": {\n                                                "_type": "VarRef",\n                                                "name": "t"\n                                            },\n                                            "expr": {\n                                                "_type": "Literal",\n                                                "value": 0.0\n                                            }\n                                        }\n                                    ],\n                                    "domain_constraint": {\n                                        "_type": "BinaryFormulaOp",\n                                        "op": "&",\n                                        "left": {\n                                            "_type": "BinaryFormulaOp",\n                                            "op": "&",\n                                            "left": {\n                                                "_type": "Comparison",\n                                                "op": "<=",\n                                                "left": {\n                                                    "_type": "Literal",\n                                                    "value": 15.0\n                                                },\n                                                "right": {\n                                                    "_type": "VarRef",\n                                                    "name": "v"\n                                                }\n                                            },\n                                            "right": {\n                                                "_type": "Comparison",\n                                                "op": "<=",\n                                                "left": {\n                                                    "_type": "VarRef",\n                                                    "name": "v"\n                                                },\n                                                "right": {\n                                                    "_type": "Literal",\n                                                    "value": 40.0\n                                                }\n                                            }\n                                        },\n                                        "right": {\n                                            "_type": "Comparison",\n                                            "op": "=",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "t"\n                                            },\n                                            "right": {\n                                                "_type": "Literal",\n                                                "value": 2.5\n                                            }\n                                        }\n                                    }\n                                }\n                            ]\n                        },\n                        "right": {\n                            "_type": "Choice",\n                            "left": {\n                                "_type": "Sequence",\n                                "programs": [\n                                    {\n                                        "_type": "Test",\n                                        "condition": {\n                                            "_type": "Comparison",\n                                            "op": "=",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "mode"\n                                            },\n                                            "right": {\n                                                "_type": "FunctionCall",\n                                                "name": "sbrakeact",\n                                                "args": []\n                                            }\n                                        }\n                                    },\n                                    {\n                                        "_type": "ODE",\n                                        "equations": [\n                                            {\n                                                "_type": "ODEEquation",\n                                                "var": {\n                                                    "_type": "VarRef",\n                                                    "name": "v"\n                                                },\n                                                "expr": {\n                                                    "_type": "BinaryTermOp",\n                                                    "op": "-",\n                                                    "left": {\n                                                        "_type": "UnaryTermOp",\n                                                        "op": "-",\n                                                        "operand": {\n                                                            "_type": "VarRef",\n                                                            "name": "t"\n                                                        }\n                                                    },\n                                                    "right": {\n                                                        "_type": "Literal",\n                                                        "value": 1.2\n                                                    }\n                                                }\n                                            },\n                                            {\n                                                "_type": "ODEEquation",\n                                                "var": {\n                                                    "_type": "VarRef",\n                                                    "name": "x"\n                                                },\n                                                "expr": {\n                                                    "_type": "Literal",\n                                                    "value": 0.0\n                                                }\n                                            },\n                                            {\n                                                "_type": "ODEEquation",\n                                                "var": {\n                                                    "_type": "VarRef",\n                                                    "name": "t"\n                                                },\n                                                "expr": {\n                                                    "_type": "Literal",\n                                                    "value": 0.5\n                                                }\n                                            }\n                                        ],\n                                        "domain_constraint": {\n                                            "_type": "BinaryFormulaOp",\n                                            "op": "&",\n                                            "left": {\n                                                "_type": "BinaryFormulaOp",\n                                                "op": "&",\n                                                "left": {\n                                                    "_type": "BinaryFormulaOp",\n                                                    "op": "&",\n                                                    "left": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "Literal",\n                                                            "value": 5.0\n                                                        },\n                                                        "right": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        }\n                                                    },\n                                                    "right": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        },\n                                                        "right": {\n                                                            "_type": "Literal",\n                                                            "value": 20.0\n                                                        }\n                                                    }\n                                                },\n                                                "right": {\n                                                    "_type": "Comparison",\n                                                    "op": "<=",\n                                                    "left": {\n                                                        "_type": "Literal",\n                                                        "value": 0.0\n                                                    },\n                                                    "right": {\n                                                        "_type": "VarRef",\n                                                        "name": "t"\n                                                    }\n                                                }\n                                            },\n                                            "right": {\n                                                "_type": "Comparison",\n                                                "op": "<=",\n                                                "left": {\n                                                    "_type": "VarRef",\n                                                    "name": "t"\n                                                },\n                                                "right": {\n                                                    "_type": "Literal",\n                                                    "value": 1.3\n                                                }\n                                            }\n                                        }\n                                    }\n                                ]\n                            },\n                            "right": {\n                                "_type": "Choice",\n                                "left": {\n                                    "_type": "Sequence",\n                                    "programs": [\n                                        {\n                                            "_type": "Test",\n                                            "condition": {\n                                                "_type": "Comparison",\n                                                "op": "=",\n                                                "left": {\n                                                    "_type": "VarRef",\n                                                    "name": "mode"\n                                                },\n                                                "right": {\n                                                    "_type": "FunctionCall",\n                                                    "name": "sbrakefull",\n                                                    "args": []\n                                                }\n                                            }\n                                        },\n                                        {\n                                            "_type": "ODE",\n                                            "equations": [\n                                                {\n                                                    "_type": "ODEEquation",\n                                                    "var": {\n                                                        "_type": "VarRef",\n                                                        "name": "v"\n                                                    },\n                                                    "expr": {\n                                                        "_type": "UnaryTermOp",\n                                                        "op": "-",\n                                                        "operand": {\n                                                            "_type": "Literal",\n                                                            "value": 2.5\n                                                        }\n                                                    }\n                                                },\n                                                {\n                                                    "_type": "ODEEquation",\n                                                    "var": {\n                                                        "_type": "VarRef",\n                                                        "name": "x"\n                                                    },\n                                                    "expr": {\n                                                        "_type": "Literal",\n                                                        "value": 0.0\n                                                    }\n                                                },\n                                                {\n                                                    "_type": "ODEEquation",\n                                                    "var": {\n                                                        "_type": "VarRef",\n                                                        "name": "t"\n                                                    },\n                                                    "expr": {\n                                                        "_type": "Literal",\n                                                        "value": 0.0\n                                                    }\n                                                }\n                                            ],\n                                            "domain_constraint": {\n                                                "_type": "BinaryFormulaOp",\n                                                "op": "&",\n                                                "left": {\n                                                    "_type": "BinaryFormulaOp",\n                                                    "op": "&",\n                                                    "left": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "Literal",\n                                                            "value": 5.0\n                                                        },\n                                                        "right": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        }\n                                                    },\n                                                    "right": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        },\n                                                        "right": {\n                                                            "_type": "Literal",\n                                                            "value": 20.0\n                                                        }\n                                                    }\n                                                },\n                                                "right": {\n                                                    "_type": "Comparison",\n                                                    "op": "=",\n                                                    "left": {\n                                                        "_type": "VarRef",\n                                                        "name": "t"\n                                                    },\n                                                    "right": {\n                                                        "_type": "Literal",\n                                                        "value": 1.3\n                                                    }\n                                                }\n                                            }\n                                        }\n                                    ]\n                                },\n                                "right": {\n                                    "_type": "Choice",\n                                    "left": {\n                                        "_type": "Sequence",\n                                        "programs": [\n                                            {\n                                                "_type": "Test",\n                                                "condition": {\n                                                    "_type": "Comparison",\n                                                    "op": "=",\n                                                    "left": {\n                                                        "_type": "VarRef",\n                                                        "name": "mode"\n                                                    },\n                                                    "right": {\n                                                        "_type": "FunctionCall",\n                                                        "name": "normalPI",\n                                                        "args": []\n                                                    }\n                                                }\n                                            },\n                                            {\n                                                "_type": "ODE",\n                                                "equations": [\n                                                    {\n                                                        "_type": "ODEEquation",\n                                                        "var": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        },\n                                                        "expr": {\n                                                            "_type": "BinaryTermOp",\n                                                            "op": "-",\n                                                            "left": {\n                                                                "_type": "BinaryTermOp",\n                                                                "op": "*",\n                                                                "left": {\n                                                                    "_type": "UnaryTermOp",\n                                                                    "op": "-",\n                                                                    "operand": {\n                                                                        "_type": "Literal",\n                                                                        "value": 0.001\n                                                                    }\n                                                                },\n                                                                "right": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "x"\n                                                                }\n                                                            },\n                                                            "right": {\n                                                                "_type": "BinaryTermOp",\n                                                                "op": "*",\n                                                                "left": {\n                                                                    "_type": "Literal",\n                                                                    "value": 0.052\n                                                                },\n                                                                "right": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "v"\n                                                                }\n                                                            }\n                                                        }\n                                                    },\n                                                    {\n                                                        "_type": "ODEEquation",\n                                                        "var": {\n                                                            "_type": "VarRef",\n                                                            "name": "x"\n                                                        },\n                                                        "expr": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        }\n                                                    },\n                                                    {\n                                                        "_type": "ODEEquation",\n                                                        "var": {\n                                                            "_type": "VarRef",\n                                                            "name": "t"\n                                                        },\n                                                        "expr": {\n                                                            "_type": "Literal",\n                                                            "value": 0.0\n                                                        }\n                                                    }\n                                                ],\n                                                "domain_constraint": {\n                                                    "_type": "BinaryFormulaOp",\n                                                    "op": "&",\n                                                    "left": {\n                                                        "_type": "BinaryFormulaOp",\n                                                        "op": "&",\n                                                        "left": {\n                                                            "_type": "BinaryFormulaOp",\n                                                            "op": "&",\n                                                            "left": {\n                                                                "_type": "Comparison",\n                                                                "op": "<=",\n                                                                "left": {\n                                                                    "_type": "UnaryTermOp",\n                                                                    "op": "-",\n                                                                    "operand": {\n                                                                        "_type": "Literal",\n                                                                        "value": 15.0\n                                                                    }\n                                                                },\n                                                                "right": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "v"\n                                                                }\n                                                            },\n                                                            "right": {\n                                                                "_type": "Comparison",\n                                                                "op": "<=",\n                                                                "left": {\n                                                                    "_type": "VarRef",\n                                                                    "name": "v"\n                                                                },\n                                                                "right": {\n                                                                    "_type": "Literal",\n                                                                    "value": 15.0\n                                                                }\n                                                            }\n                                                        },\n                                                        "right": {\n                                                            "_type": "Comparison",\n                                                            "op": "<=",\n                                                            "left": {\n                                                                "_type": "UnaryTermOp",\n                                                                "op": "-",\n                                                                "operand": {\n                                                                    "_type": "Literal",\n                                                                    "value": 500.0\n                                                                }\n                                                            },\n                                                            "right": {\n                                                                "_type": "VarRef",\n                                                                "name": "x"\n                                                            }\n                                                        }\n                                                    },\n                                                    "right": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "VarRef",\n                                                            "name": "x"\n                                                        },\n                                                        "right": {\n                                                            "_type": "Literal",\n                                                            "value": 500.0\n                                                        }\n                                                    }\n                                                }\n                                            }\n                                        ]\n                                    },\n                                    "right": {\n                                        "_type": "Sequence",\n                                        "programs": [\n                                            {\n                                                "_type": "Test",\n                                                "condition": {\n                                                    "_type": "Comparison",\n                                                    "op": "=",\n                                                    "left": {\n                                                        "_type": "VarRef",\n                                                        "name": "mode"\n                                                    },\n                                                    "right": {\n                                                        "_type": "FunctionCall",\n                                                        "name": "accelerate",\n                                                        "args": []\n                                                    }\n                                                }\n                                            },\n                                            {\n                                                "_type": "ODE",\n                                                "equations": [\n                                                    {\n                                                        "_type": "ODEEquation",\n                                                        "var": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        },\n                                                        "expr": {\n                                                            "_type": "Literal",\n                                                            "value": 1.5\n                                                        }\n                                                    },\n                                                    {\n                                                        "_type": "ODEEquation",\n                                                        "var": {\n                                                            "_type": "VarRef",\n                                                            "name": "x"\n                                                        },\n                                                        "expr": {\n                                                            "_type": "Literal",\n                                                            "value": 0.0\n                                                        }\n                                                    },\n                                                    {\n                                                        "_type": "ODEEquation",\n                                                        "var": {\n                                                            "_type": "VarRef",\n                                                            "name": "t"\n                                                        },\n                                                        "expr": {\n                                                            "_type": "Literal",\n                                                            "value": 0.0\n                                                        }\n                                                    }\n                                                ],\n                                                "domain_constraint": {\n                                                    "_type": "BinaryFormulaOp",\n                                                    "op": "&",\n                                                    "left": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "UnaryTermOp",\n                                                            "op": "-",\n                                                            "operand": {\n                                                                "_type": "Literal",\n                                                                "value": 20.0\n                                                            }\n                                                        },\n                                                        "right": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        }\n                                                    },\n                                                    "right": {\n                                                        "_type": "Comparison",\n                                                        "op": "<=",\n                                                        "left": {\n                                                            "_type": "VarRef",\n                                                            "name": "v"\n                                                        },\n                                                        "right": {\n                                                            "_type": "UnaryTermOp",\n                                                            "op": "-",\n                                                            "operand": {\n                                                                "_type": "Literal",\n                                                                "value": 5.0\n                                                            }\n                                                        }\n                                                    }\n                                                }\n                                            }\n                                        ]\n                                    }\n                                }\n                            }\n                        }\n                    }\n                }\n            ]\n        },\n        "verification_formula_ast_json": {\n            "_type": "Quantifier",\n            "quantifier_type": "forall",\n            "variables": [\n                {\n                    "_type": "VarRef",\n                    "name": "eps"\n                }\n            ],\n            "body": {\n                "_type": "BinaryFormulaOp",\n                "op": "->",\n                "left": {\n                    "_type": "Comparison",\n                    "op": ">",\n                    "left": {\n                        "_type": "VarRef",\n                        "name": "eps"\n                    },\n                    "right": {\n                        "_type": "Literal",\n                        "value": 0.0\n                    }\n                },\n                "right": {\n                    "_type": "Quantifier",\n                    "quantifier_type": "exists",\n                    "variables": [\n                        {\n                            "_type": "VarRef",\n                            "name": "del"\n                        }\n                    ],\n                    "body": {\n                        "_type": "BinaryFormulaOp",\n                        "op": "&",\n                        "left": {\n                            "_type": "Comparison",\n                            "op": ">",\n                            "left": {\n                                "_type": "VarRef",\n                                "name": "del"\n                            },\n                            "right": {\n                                "_type": "Literal",\n                                "value": 0.0\n                            }\n                        },\n                        "right": {\n                            "_type": "Quantifier",\n                            "quantifier_type": "forall",\n                            "variables": [\n                                {\n                                    "_type": "VarRef",\n                                    "name": "v"\n                                }\n                            ],\n                            "body": {\n                                "_type": "BinaryFormulaOp",\n                                "op": "->",\n                                "left": {\n                                    "_type": "Comparison",\n                                    "op": "<",\n                                    "left": {\n                                        "_type": "BinaryTermOp",\n                                        "op": "^",\n                                        "left": {\n                                            "_type": "VarRef",\n                                            "name": "v"\n                                        },\n                                        "right": {\n                                            "_type": "Literal",\n                                            "value": 2.0\n                                        }\n                                    },\n                                    "right": {\n                                        "_type": "BinaryTermOp",\n                                        "op": "^",\n                                        "left": {\n                                            "_type": "VarRef",\n                                            "name": "del"\n                                        },\n                                        "right": {\n                                            "_type": "Literal",\n                                            "value": 2.0\n                                        }\n                                    }\n                                },\n                                "right": {\n                                    "_type": "ModalFormula",\n                                    "modality": "box",\n                                    "program": {\n                                        "_type": "VarRef",\n                                        "name": "cruiseControl"\n                                    },\n                                    "formula": {\n                                        "_type": "Comparison",\n                                        "op": "<",\n                                        "left": {\n                                            "_type": "BinaryTermOp",\n                                            "op": "^",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "v"\n                                            },\n                                            "right": {\n                                                "_type": "Literal",\n                                                "value": 2.0\n                                            }\n                                        },\n                                        "right": {\n                                            "_type": "BinaryTermOp",\n                                            "op": "^",\n                                            "left": {\n                                                "_type": "VarRef",\n                                                "name": "eps"\n                                            },\n                                            "right": {\n                                                "_type": "Literal",\n                                                "value": 2.0\n                                            }\n                                        }\n                                    }\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n        },\n        "simulation_reward_terms": [\n            "((0.0) + ((0.0) if (self._cruiseSafe_impl()) else (-100)))",\n            "((-(abs(v))) + ((10) if (self._nearZeroSpeed_impl()) else (0.0)))"\n        ],\n        "simulation_predicate_rules": [\n            {\n                "predicate": "self._cruiseSafe_impl()",\n                "kind": "safety",\n                "on_success": "continue",\n                "success_status": "none",\n                "on_failure": "truncate",\n                "failure_status": "failure"\n            },\n            {\n                "predicate": "self._nearZeroSpeed_impl()",\n                "kind": "progress",\n                "on_success": "continue",\n                "success_status": "none",\n                "on_failure": "continue",\n                "failure_status": "none"\n            }\n        ]\n    }'


class KeYmaeraGymEnv(gym.Env):
    metadata = {
    "render_modes": [
        "human",
        "rgb_array"
    ],
    "render_fps": 30
}

    def __init__(self, render_mode=None, config_override=None):
        super().__init__()
        self.config = self._load_config(_ENV_CONFIG_JSON, config_override)
        
        self.debug_mode = self.config.get('debug_mode', False)
        
        # DEBUG PRINTS START
        self._debug_print(f"Initialized self.config['variables']: {self.config['variables']}")
        self._debug_print(f"Initialized self.config['constants']: {self.config['constants']}")
        # DEBUG PRINTS END

        self.environment_name = self.config['environment_name']
        self.render_mode = render_mode
        self.current_state = OrderedDict()
        self.time = 0.0 # Time elapsed within current control step (t in dL)
        self.simulation_elapsed_time = 0.0
        self.global_time_remaining = 0.0 # T in dL
        self.current_step_count = 0 # Number of control steps taken
        self.max_episode_steps = self.config.get('max_episode_steps', 200)
        
        # Default time_step, will be overridden by 'ep' constant if present
        self.time_step = self.config.get('time_step', 0.01) 
        
        self.ode_solver_method = self.config.get('ode_solver_method', 'dopri5')
        self.ode_solver_step_size = self.config.get('ode_solver_step_size', 0.001)
        self.ode_solver_tolerance = self.config.get('ode_solver_tolerance', 1e-6)

        # SymPy symbols for dynamic evaluation of ODEs
        self.sympy_symbols = {}
        # Variables and constants need to be SymPy Symbols for ODE parsing
        for var_config in self.config['variables']:
            self._debug_print(f"DEBUG: Initializing sympy_symbols with variable: {var_config['name']}")
            self.sympy_symbols[var_config['name']] = sympy.Symbol(var_config['name'])
        for const_config in self.config['constants']:
            self._debug_print(f"DEBUG: Initializing sympy_symbols with constant: {const_config['name']}")
            self.sympy_symbols[const_config['name']] = sympy.Symbol(const_config['name'])

        # Initialize user_defined_functions before _setup_constants calls _evaluate_expression
        self.user_defined_functions = {} 

        self._resolve_constants_with_solver()
        self._setup_constants()
        self._setup_functions() 

        ep_value = None
        # Check if 'ep' is a constant
        ep_const_config = next((c for c in self.config['constants'] if c['name'] == 'ep'), None)
        if ep_const_config:
            # If it's a constant, its value is already in self.current_state from _setup_constants
            ep_value = self.current_state['ep'] 
        else: # Check if 'ep' is a function without arguments
            ep_func_config = next((f for f in self.config['function_definitions'] if f['name'] == 'ep' and not f['args']), None)
            if ep_func_config:
                ep_value = self.user_defined_functions['ep']() # Call the generated _ep_impl method

        if ep_value is not None:
            self.time_step = ep_value

        self.control_flow_program_ast: Optional[HybridProgram] = None
        if self.config.get('control_flow_program_ast_json'):
            self.control_flow_program_ast = self._deserialize_ast_node(self.config['control_flow_program_ast_json'])
            self._debug_print(f"DEBUG: Deserialized control_flow_program_ast: {self.control_flow_program_ast._repr_indented(0)}")

        self.simulation_mode = self.config.get('simulation_mode', False)
        self.simulation_initialization_ast = (
            self._deserialize_ast_node(self.config['simulation_initialization_ast_json'])
            if self.config.get('simulation_initialization_ast_json') else None
        )
        self.simulation_step_ast = (
            self._deserialize_ast_node(self.config['simulation_step_ast_json'])
            if self.config.get('simulation_step_ast_json') else None
        )
        self._simulation_expression_cache = {}
        self.control_choice_paths = (
            self._enumerate_control_choice_paths(self.control_flow_program_ast)
            if self.control_flow_program_ast else []
        )
        self._setup_spaces()
        self._setup_initial_state_samplers()
        self._setup_hybrid_program()
        
        # Initial State Buffer (ISB) setup
        self.initial_state_buffer_config = self.config.get('initial_state_buffer_config', {})
        self.initial_state_buffer_strategy = self.initial_state_buffer_config.get('strategy', 'random')
        self.initial_state_buffer = deque(maxlen=self.initial_state_buffer_config.get('capacity', 100))
        
        if self.initial_state_buffer_config.get('enabled', False) and self.initial_state_buffer_config.get('populate_on_init', True):
            self._populate_initial_state_buffer()
        
    def _debug_print(self, message):
        if self.debug_mode:
            print(f"DEBUG: {message}")

    # User-defined functions from KeYmaera model
    def _accelerate_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("0.0", context)

    def _ebrakeact_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("1.0", context)

    def _ebrakefull_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("2.0", context)

    def _normalPI_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("3.0", context)

    def _sbrakeact_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("4.0", context)

    def _sbrakefull_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("5.0", context)

    def _cruiseSafe_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("((((((((-(500.0) <= x) and (x <= 500.0)) and (-(20.0) <= v)) and (v <= 40.0)) and (0.0 <= t)) and (t <= 2.5)) and (0.0 <= mode)) and (mode <= 5.0))", context)

    def _nearZeroSpeed_impl(self):
        context = {}
        arg_names = []
        for name, val in zip(arg_names, []):
            context[name] = val
        return self._evaluate_expression("(abs(v) <= 1.0)", context)


    # --- AST Deserialization ---
    def _deserialize_ast_node(self, data: Dict[str, Any]) -> ASTNode:
        if data is None:
            return None
        
        node_type = data["_type"]
        if node_type == "Literal":
            return Literal(data["value"])
        elif node_type == "VarRef":
            return VarRef(data["name"])
        elif node_type == "BinaryTermOp":
            return BinaryTermOp(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "UnaryTermOp":
            return UnaryTermOp(data["op"], self._deserialize_ast_node(data["operand"]))
        elif node_type == "FunctionCall":
            return FunctionCall(data["name"], [self._deserialize_ast_node(arg) for arg in data["args"]])
        elif node_type == "BooleanLiteral":
            return BooleanLiteral(data["value"])
        elif node_type == "BinaryFormulaOp":
            return BinaryFormulaOp(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "UnaryFormulaOp":
            return UnaryFormulaOp(data["op"], self._deserialize_ast_node(data["operand"]))
        elif node_type == "Comparison":
            return Comparison(data["op"], self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "PredicateCall":
            return PredicateCall(data["name"], [self._deserialize_ast_node(arg) for arg in data["args"]])
        elif node_type == "Assignment":
            return Assignment(self._deserialize_ast_node(data["var"]), self._deserialize_ast_node(data["expr"]))
        elif node_type == "NondetAssignment":
            return NondetAssignment(self._deserialize_ast_node(data["var"]))
        elif node_type == "Test":
            return Test(self._deserialize_ast_node(data["condition"]))
        elif node_type == "Sequence":
            return Sequence([self._deserialize_ast_node(p) for p in data["programs"]])
        elif node_type == "Choice":
            return Choice(self._deserialize_ast_node(data["left"]), self._deserialize_ast_node(data["right"]))
        elif node_type == "Iteration":
            return Iteration(self._deserialize_ast_node(data["program"]), self._deserialize_ast_node(data["invariant"]))
        elif node_type == "ODE":
            return ODE([self._deserialize_ast_node(eq) for eq in data["equations"]], self._deserialize_ast_node(data["domain_constraint"]))
        elif node_type == "ODEEquation":
            return ODEEquation(self._deserialize_ast_node(data["var"]), self._deserialize_ast_node(data["expr"]))
        elif node_type == "HPBlock":
            return HPBlock(self._deserialize_ast_node(data["program"]))
        elif node_type == "ModalFormula":
            return ModalFormula(data["modality"], self._deserialize_ast_node(data["program"]), self._deserialize_ast_node(data["formula"]))
        elif node_type == "Quantifier":
            return Quantifier(data["quantifier_type"], [self._deserialize_ast_node(v) for v in data["variables"]], self._deserialize_ast_node(data["body"]))
        elif node_type == "C_Formula":
            return C_Formula(self._deserialize_ast_node(data["formula"]))
        else:
            raise ValueError(f"Unknown AST node type for deserialization: {node_type}")

    # --- Python Eval String Conversion for Environment's ASTs ---
    _PYTHON_TERM_OPS = {'+': '+', '-': '-', '*': '*', '/': '/', '^': '**'}
    _PYTHON_FORMULA_OPS = {
        '=': '==', '!=': '!=', '>': '>', '<': '<', '>=': '>=', '<=': '<=',
        '&&': 'and', '&': 'and', '|': 'or'
    }

    def _term_to_python_eval_str_env(self, term: ASTNode) -> str:
        """Converts an AST Term node to its string representation for Python eval() within the environment."""
        if isinstance(term, Literal):
            return str(term.value)
        elif isinstance(term, VarRef):
            # Access variable from current_state
            return f"self.current_state['{term.name}']"
        elif isinstance(term, BinaryTermOp):
            left_str = self._term_to_python_eval_str_env(term.left)
            right_str = self._term_to_python_eval_str_env(term.right)
            op = self._PYTHON_TERM_OPS.get(term.op, term.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(term, UnaryTermOp):
            operand_str = self._term_to_python_eval_str_env(term.operand)
            if term.op == "'":
                raise ValueError(f"Derivative operator '{term.op}' not supported in Python term evaluation in environment: {term}")
            return f"{term.op}({operand_str})"
        elif isinstance(term, FunctionCall):
            args_str = ", ".join(self._term_to_python_eval_str_env(arg) for arg in term.args)
            if term.name in {'abs', 'min', 'max'}:
                return f"{term.name}({args_str})"
            # Call the bound method from user_defined_functions
            return f"self.user_defined_functions['{term.name}']({args_str})"
        else:
            raise ValueError(f"Unsupported Term type for Python eval in environment: {type(term)}")

    def _formula_to_python_eval_str_env(self, formula: ASTNode) -> str:
        """Converts an AST Formula node to its string representation for Python eval() within the environment."""
        if isinstance(formula, BooleanLiteral):
            return str(formula.value)
        elif isinstance(formula, Comparison):
            left_str = self._term_to_python_eval_str_env(formula.left)
            right_str = self._term_to_python_eval_str_env(formula.right)
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, BinaryFormulaOp):
            left_str = self._formula_to_python_eval_str_env(formula.left)
            right_str = self._formula_to_python_eval_str_env(formula.right)
            if formula.op == "->":
                return f"(not ({left_str}) or ({right_str}))"
            elif formula.op == "<->":
                return f"((({left_str}) and ({right_str})) or (not ({left_str}) and not ({right_str})))"
            op = self._PYTHON_FORMULA_OPS.get(formula.op, formula.op)
            return f"({left_str} {op} {right_str})"
        elif isinstance(formula, UnaryFormulaOp):
            operand_str = self._formula_to_python_eval_str_env(formula.operand)
            if formula.op == "!":
                return f"(not ({operand_str}))"
            return f"{formula.op}({operand_str})"
        elif isinstance(formula, PredicateCall):
            args_str = ", ".join(self._term_to_python_eval_str_env(arg) for arg in formula.args)
            return f"self.user_defined_functions['{formula.name}']({args_str})"
        elif isinstance(formula, (ModalFormula, Quantifier, C_Formula)):
            raise ValueError(f"Formula type {type(formula)} cannot be directly converted to Python for evaluation in environment: {formula}")
        else:
            raise ValueError(f"Unsupported Formula type for Python eval in environment: {type(formula)}")


    def _load_config(self, config_json_str, config_override):
        config = json.loads(config_json_str)
        if config_override:
            for key, override_value in config_override.items():
                if key == 'constants' and isinstance(override_value, list):
                    # Merge constants list by name
                    existing_constants_map = {c['name']: c for c in config.get('constants', [])}
                    for ov_const in override_value:
                        if ov_const['name'] in existing_constants_map:
                            if 'value' in ov_const and not {
                                'lower_bound', 'upper_bound'
                            }.intersection(ov_const):
                                existing_constants_map[ov_const['name']].pop('lower_bound', None)
                                existing_constants_map[ov_const['name']].pop('upper_bound', None)
                            existing_constants_map[ov_const['name']].update(ov_const)
                        else:
                            existing_constants_map[ov_const['name']] = ov_const # Add new constant
                    config['constants'] = list(existing_constants_map.values())
                elif key == 'variables' and isinstance(override_value, list):
                    # Merge variables list by name
                    existing_variables_map = {v['name']: v for v in config.get('variables', [])}
                    for ov_var in override_value:
                        if ov_var['name'] in existing_variables_map:
                            existing_variables_map[ov_var['name']].update(ov_var)
                        else:
                            existing_variables_map[ov_var['name']] = ov_var # Add new variable
                    config['variables'] = list(existing_variables_map.values())
                elif key in config and isinstance(config[key], dict) and isinstance(override_value, dict):
                    # For nested dictionaries, update them
                    config[key].update(override_value)
                else:
                    # For other keys, just replace the value
                    config[key] = override_value
        return config

    def _resolve_constants_with_solver(self):
        constants = self.config.get('constants', [])
        ranged_constants = [
            constant for constant in constants
            if constant.get('lower_bound') is not None
            and constant.get('upper_bound') is not None
        ]
        has_bounds_predicate = any(
            function.get('name') == 'bounds'
            for function in self.config.get('function_definitions', [])
        )
        if not ranged_constants and not has_bounds_predicate:
            return

        z3_symbols = {
            constant['name']: z3.Real(constant['name'])
            for constant in ranged_constants
        }
        fixed_constants_map = {}
        for constant in constants:
            name = constant['name']
            if name in z3_symbols:
                continue
            try:
                fixed_constants_map[name] = float(constant['value'])
            except (TypeError, ValueError):
                continue

        solver = z3.Solver()
        for constant in ranged_constants:
            symbol = z3_symbols[constant['name']]
            low = float(constant['lower_bound'])
            high = float(constant['upper_bound'])
            if low > high:
                raise ValueError(
                    f"Invalid @const range for {constant['name']}: [{low}, {high}]"
                )
            solver.add(symbol >= low, symbol <= high)

        if has_bounds_predicate:
            bounds_condition = self._expr_to_z3(
                'self._bounds_impl()', z3_symbols, fixed_constants_map
            )
            solver.add(bounds_condition)

        result = solver.check()
        if result == z3.unsat:
            if has_bounds_predicate:
                fixed_validation = z3.Solver()
                fixed_validation.add(self._expr_to_z3(
                    'self._bounds_impl()', {}, fixed_constants_map
                ))
                fixed_result = fixed_validation.check()
                if fixed_result == z3.unsat:
                    raise ValueError(
                        "Fixed @const values do not satisfy the model bounds() predicate."
                    )
                if fixed_result != z3.sat:
                    raise RuntimeError(
                        "Z3 could not validate fixed constants against bounds(): "
                        f"{fixed_validation.reason_unknown()}"
                    )
            constraint_description = (
                "the @const ranges and model bounds() predicate"
                if has_bounds_predicate else "the @const ranges"
            )
            raise ValueError(
                f"No constant assignment satisfies {constraint_description}."
            )
        if result != z3.sat:
            raise RuntimeError(
                f"Z3 could not resolve model constants: {solver.reason_unknown()}"
            )

        model = solver.model()
        for constant in ranged_constants:
            value = model.evaluate(
                z3_symbols[constant['name']], model_completion=True
            )
            constant['value'] = str(value)

        if has_bounds_predicate:
            concrete_constants = {}
            for constant in constants:
                try:
                    concrete_constants[constant['name']] = float(constant['value'])
                except (TypeError, ValueError):
                    continue
            validation = z3.Solver()
            validation.add(self._expr_to_z3(
                'self._bounds_impl()', {}, concrete_constants
            ))
            validation_result = validation.check()
            if validation_result == z3.unsat:
                raise ValueError(
                    "Fixed @const values do not satisfy the model bounds() predicate."
                )
            if validation_result != z3.sat:
                raise RuntimeError(
                    "Z3 could not validate fixed constants against bounds(): "
                    f"{validation.reason_unknown()}"
                )

    def _setup_spaces(self):
        # Give the agent an opaque ID for a complete path through the control AST.
        if self.control_flow_program_ast:
            num_choices = len(self.control_choice_paths)
            
            if num_choices > 0:
                self.action_space_type = 'discrete'
                self.action_space = spaces.Discrete(num_choices)
                self.action_variable_names = ['control_choice']
                self._debug_print(f"DEBUG: Action space set to Discrete({num_choices}) based on complete control paths.")
            else:
                raise ValueError("controlFunc is a Choice but no valid branches found.")
        elif self.control_flow_program_ast:
            # If controlFunc is present but not a Choice, it's a single, fixed discrete action.
            # Or it could be a NondetAssignment, which is more complex.
            # For now, if it's not a Choice, assume it's a single "action" that executes the HP.
            self.action_space_type = 'discrete'
            self.action_space = spaces.Discrete(1)
            self.action_variable_names = ['control_choice']
            self._debug_print(f"DEBUG: Action space set to Discrete(1) based on controlFunc (not a Choice).")
        else: # Fallback to continuous if no controlFunc or it's not interpretable
            self.action_space_type = self.config.get('action_space_type', 'continuous')
            action_bounds = self.config.get('action_space_bounds', {})

            if self.action_space_type == 'continuous':
                lows = []
                highs = []
                self.action_variable_names = []
                for var_name, bounds in action_bounds.items():
                    self.action_variable_names.append(var_name)
                    lows.append(self._evaluate_expression(bounds['min'], {}))
                    highs.append(self._evaluate_expression(bounds['max'], {}))
                self.action_space = spaces.Box(low=np.array(lows, dtype=np.float32),
                                               high=np.array(highs, dtype=np.float32),
                                               dtype=np.float32)
            elif self.action_space_type == 'discrete':
                n = action_bounds.get('n', 1)
                self.action_space = spaces.Discrete(n)
                self.action_variable_names = [] # No specific variable names for generic discrete
            else:
                raise ValueError(f"Unsupported action space type: {self.action_space_type}")

        observation_space_type = self.config.get('observation_space_type', 'continuous')
        observation_bounds = self.config.get('observation_space_bounds', {})

        if observation_space_type == 'continuous':
            lows = []
            highs = []
            self.observation_variable_names = []
            for var_name, bounds in observation_bounds.items():
                self.observation_variable_names.append(var_name)
                lows.append(self._evaluate_expression(bounds['min'], {}))
                highs.append(self._evaluate_expression(bounds['max'], {}))
            self.observation_space = spaces.Box(low=np.array(lows, dtype=np.float32),
                                                high=np.array(highs, dtype=np.float32),
                                                dtype=np.float32)
        elif observation_space_type == 'discrete':
            raise NotImplementedError("Discrete observation space not yet supported.")
        else:
            raise ValueError(f"Unsupported observation space type: {observation_space_type}")

    def _enumerate_control_choice_paths(self, node):
        if isinstance(node, HPBlock):
            return self._enumerate_control_choice_paths(node.program)
        if isinstance(node, Choice):
            left_paths = self._enumerate_control_choice_paths(node.left)
            right_paths = self._enumerate_control_choice_paths(node.right)
            return [(0,) + path for path in left_paths] + [
                (1,) + path for path in right_paths
            ]
        if isinstance(node, Sequence):
            paths = [()]
            for program in node.programs:
                program_paths = self._enumerate_control_choice_paths(program)
                paths = [
                    prefix + suffix
                    for prefix in paths
                    for suffix in program_paths
                ]
            return paths
        if isinstance(node, Iteration):
            return self._enumerate_control_choice_paths(node.program)
        return [()]

    def _setup_initial_state_samplers(self):
        self.initial_state_samplers = {}
        for var_name, sampling_config in self.config['initial_state_sampling_config'].items():
            if 'value' in sampling_config:
                val = self._evaluate_expression(sampling_config['value'], {})
                self.initial_state_samplers[var_name] = lambda v=val: v
            elif 'min' in sampling_config and 'max' in sampling_config:
                min_val = self._evaluate_expression(sampling_config['min'], {})
                max_val = self._evaluate_expression(sampling_config['max'], {})
                self.initial_state_samplers[var_name] = lambda mn=min_val, mx=max_val: self.np_random.uniform(mn, mx)
            else:
                self.initial_state_samplers[var_name] = lambda: 0.0

    def _setup_constants(self):
        for const_config in self.config['constants']:
            self._debug_print(f"DEBUG: _setup_constants evaluating constant: {const_config['name']}")
            self.current_state[const_config['name']] = self._evaluate_expression(const_config['value'], {})
            # Ensure constants are also SymPy symbols if they are used in ODEs or other symbolic expressions
            if const_config['name'] not in self.sympy_symbols:
                self._debug_print(f"DEBUG: Adding constant to sympy_symbols (late): {const_config['name']}")
                self.sympy_symbols[const_config['name']] = sympy.Symbol(const_config['name'])


    def _setup_functions(self):
        self.user_defined_functions = {}
        self.user_defined_functions['accelerate'] = self._accelerate_impl

        self.user_defined_functions['ebrakeact'] = self._ebrakeact_impl

        self.user_defined_functions['ebrakefull'] = self._ebrakefull_impl

        self.user_defined_functions['normalPI'] = self._normalPI_impl

        self.user_defined_functions['sbrakeact'] = self._sbrakeact_impl

        self.user_defined_functions['sbrakefull'] = self._sbrakefull_impl

        self.user_defined_functions['cruiseSafe'] = self._cruiseSafe_impl

        self.user_defined_functions['nearZeroSpeed'] = self._nearZeroSpeed_impl



    def _setup_hybrid_program(self):
        ode_program_str = self.config['problem']['program']
        self.differential_variables = []
        self.differential_equations = {}
        self.differential_domain_constraint = None

        self._debug_print(f"DEBUG: _setup_hybrid_program: ode_program_str = '{ode_program_str}'")

        if ode_program_str.startswith('(') and ode_program_str.endswith(')'):
            ode_program_str = ode_program_str[1:-1]

        self._debug_print(f"DEBUG: _setup_hybrid_program: ode_program_str after paren removal = '{ode_program_str}'")

        # Split using the new distinct separator
        parts = ode_program_str.split('@@@DOMAIN_CONSTRAINT@@@', 1)

        self._debug_print(f"DEBUG: _setup_hybrid_program: parts = {parts}")

        equations_str = parts[0].strip() # Ensure no leading/trailing whitespace
        if len(parts) > 1:
            self.differential_domain_constraint = parts[1].strip()

        self._debug_print(f"DEBUG: _setup_hybrid_program: equations_str = '{equations_str}'")
        self._debug_print(f"DEBUG: _setup_hybrid_program: differential_domain_constraint = '{self.differential_domain_constraint}'")

        equation_parts = [p.strip() for p in equations_str.split(',') if p.strip()]
        self._debug_print(f"DEBUG: _setup_hybrid_program: equation_parts (filtered) = {equation_parts}")

        # Prepare sympy_locals for ODE parsing: symbols + symbolic functions
        ode_sympy_locals = {k: v for k, v in self.sympy_symbols.items()}
        for func_name in self.user_defined_functions: # Iterate over names of user-defined functions
            ode_sympy_locals[func_name] = sympy.Function(func_name) # Map to symbolic SymPy function

        for eq_part in equation_parts:
            if "'" in eq_part and '=' in eq_part:
                var_name_prime, expr = eq_part.split('=', 1)
                var_name = var_name_prime.replace("'", "").strip()
                expr = expr.strip()
                # Use ode_sympy_locals for sympify of ODE expressions
                self.differential_equations[var_name] = sympy.sympify(expr, locals=ode_sympy_locals)
                self.differential_variables.append(var_name)
            else:
                raise ValueError(f"Malformed ODE equation in program string: {eq_part}")

        for var in self.differential_variables:
            if var not in self.sympy_symbols:
                self.sympy_symbols[var] = sympy.Symbol(var)

    def _evaluate_expression(self, expression: Union[str, ASTNode], context: Dict[str, Any]):
        self._debug_print(f"DEBUG: _evaluate_expression called for: {expression} (type: {type(expression)})")

        if isinstance(expression, ASTNode):
            if isinstance(expression, Term):
                expression_str = self._term_to_python_eval_str_env(expression)
            elif isinstance(expression, Formula):
                expression_str = self._formula_to_python_eval_str_env(expression)
            else:
                raise ValueError(f"Unsupported AST node type for evaluation: {type(expression)}")
            self._debug_print(f"DEBUG: Converted AST to expression_str: '{expression_str}'")
        else: # Assume it's already a string
            expression_str = expression

        # Prepare the execution environment for eval()
        eval_globals = {
            '__builtins__': None, # Restrict builtins for safety
            'True': True, 'False': False, 'None': None,
            'self': self, # Allow access to self for calling methods
            'np': np, # Allow numpy functions
            'math': math, # Allow math functions (e.g., for power `**`)
            'abs': abs, 'min': min, 'max': max,
            'float': float, 'int': int, 'bool': bool,
            # Add sympy objects explicitly if any sympy constructs are still generated for eval
            'sympy': sympy,
            'Symbol': sympy.Symbol, 'Eq': sympy.Eq, 'Ne': sympy.Ne,
            'Gt': sympy.Gt, 'Lt': sympy.Lt, 'Ge': sympy.Ge, 'Le': sympy.Le,
            'And': sympy.And, 'Or': sympy.Or, 'Not': sympy.Not,
            'Implies': sympy.Implies, 'Equivalent': sympy.Equivalent,
            'Function': sympy.Function, # Needed if symbolic functions are passed to eval
        }
        
        # Add current state variables and constants to locals
        eval_locals = {}
        eval_locals.update(self.current_state)
        eval_locals.update(context) # Context variables (e.g., function arguments) - these override self.current_state values

        try:
            result = eval(expression_str, eval_globals, eval_locals)
            
            # Convert SymPy boolean results to Python booleans
            if isinstance(result, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
                return bool(result)
            # Convert SymPy numbers to Python floats
            if isinstance(result, sympy.Number):
                return float(result)
            # Ensure Python booleans are returned as bool
            if isinstance(result, (bool, np.bool_)):
                return bool(result)
            # Ensure Python numbers are returned as float
            if isinstance(result, (int, np.integer, np.floating)):
                return float(result)

            return result
        except NameError as ne:
            missing_name = str(ne).split("'")[1]
            raise RuntimeError(f"NameError: '{missing_name}' not found when evaluating expression '{expression_str}'. "
                               f"Available locals: {list(eval_locals.keys())}. Available globals: {list(eval_globals.keys())}") from ne
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression '{expression_str}': {e}")

    def _evaluate_sympy_function_calls(self, sympy_expr_node, current_subs):
        # Helper function to evaluate symbolic function calls within a SymPy expression.
        # This is necessary because user-defined functions are Python callables, not SymPy's built-in functions.
        # Base case 1: If it's a number (sympy.Float or sympy.Integer), return its Python float value.
        # This handles direct numerical constants.
        if isinstance(sympy_expr_node, (sympy.Float, sympy.Integer)):
            return float(sympy_expr_node)

        # Base case 2: If it's a simple sympy.Symbol, substitute its value directly.
        # This must come BEFORE the `hasattr(sympy_expr_node, 'args')` check to prevent the Symbol() TypeError.
        if isinstance(sympy_expr_node, sympy.Symbol):
            # If the symbol has a numerical value in the substitution dictionary, use it.
            if sympy_expr_node in current_subs:
                return current_subs[sympy_expr_node]
            # If not in subs_dict, it means it's a symbol that should remain symbolic.
            # In contexts like ODE evaluation, all variables should ideally be numerical at this point.
            # Returning the symbol itself might lead to further errors if the solver expects only numbers.
            return sympy_expr_node

        # If it's a SymPy Function and its name matches a user-defined function
        if isinstance(sympy_expr_node, sympy.Function) and str(sympy_expr_node.func) in self.user_defined_functions:
            func_name = str(sympy_expr_node.func)
            # Recursively evaluate arguments before calling the Python function
            evaluated_args = [self._evaluate_sympy_function_calls(arg, current_subs) for arg in sympy_expr_node.args]
            # Call the actual Python implementation method (which uses _evaluate_expression)
            # self.user_defined_functions stores the bound methods, so no need to pass 'self' explicitly.
            return self.user_defined_functions[func_name](*evaluated_args) 
        
        # Recursively apply to arguments of other SymPy expressions (e.g., Add, Mul, Pow, etc.)
        # This check should come after handling specific base types like Symbol, Float, Integer.
        if hasattr(sympy_expr_node, 'args'): # Check if it's a SymPy expression with arguments
            # Create a new expression with evaluated arguments
            return sympy_expr_node.func(*[self._evaluate_sympy_function_calls(arg, current_subs) for arg in sympy_expr_node.args])
        
        # Fallback: If it's none of the above (e.g., a Python literal like an int/float that wasn't part of a SymPy expression,
        # or a SymPy object that doesn't have args and wasn't caught by earlier checks like sympy.Number),
        # attempt a final substitution or just return the node.
        # The original `sympy_expr_node.subs(current_subs)` is generally robust for SymPy objects.
        try:
            return sympy_expr_node.subs(current_subs)
        except AttributeError:
            # If it's a pure Python object that doesn't have a .subs method, return it as is.
            return sympy_expr_node

    def _simulation_domain_comparisons(self, formula):
        if isinstance(formula, BinaryFormulaOp) and formula.op in ('&', '&&'):
            return (
                self._simulation_domain_comparisons(formula.left)
                + self._simulation_domain_comparisons(formula.right)
            )
        if isinstance(formula, Comparison):
            return [formula]
        return []

    def _compile_simulation_term(self, term):
        cache_key = id(term)
        if cache_key not in self._simulation_expression_cache:
            expression = self._term_to_python_eval_str_env(term)
            self._simulation_expression_cache[cache_key] = compile(
                expression, "<hybrid-program-term>", "eval"
            )
        return self._simulation_expression_cache[cache_key]

    def _flatten_simulation_choice(self, choice):
        if isinstance(choice, Choice):
            return (
                self._flatten_simulation_choice(choice.left)
                + self._flatten_simulation_choice(choice.right)
            )
        return [choice]

    def _integrate_simulation_ode(self, ode):
        if ode.domain_constraint is not None and not self._evaluate_expression(
            ode.domain_constraint, self.current_state
        ):
            return False

        if ode.domain_constraint is not None:
            snap_tolerance = max(1e-9, self.ode_solver_tolerance * 10)
            for comparison in self._simulation_domain_comparisons(ode.domain_constraint):
                if not isinstance(comparison.left, VarRef) and not isinstance(
                    comparison.right, VarRef
                ):
                    continue
                left_value = float(self._evaluate_expression(comparison.left, self.current_state))
                right_value = float(self._evaluate_expression(comparison.right, self.current_state))
                if abs(left_value - right_value) > snap_tolerance:
                    continue
                if isinstance(comparison.left, VarRef):
                    self.current_state[comparison.left.name] = right_value
                else:
                    self.current_state[comparison.right.name] = left_value

        variables = [equation.var.name for equation in ode.equations]
        initial_values = [float(self.current_state[name]) for name in variables]
        initial_state = dict(self.current_state)
        eval_globals = {
            "__builtins__": None,
            "self": self,
            "np": np,
            "math": math,
            "abs": abs,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "bool": bool,
        }
        derivative_codes = [
            self._compile_simulation_term(equation.expr) for equation in ode.equations
        ]

        def state_at(values):
            state = dict(initial_state)
            state.update(zip(variables, values))
            return state

        def evaluate_at_state(code, state):
            previous_state = self.current_state
            self.current_state = state
            try:
                return eval(code, eval_globals, state)
            finally:
                self.current_state = previous_state

        def derivatives(_time, values):
            state = state_at(values)
            return [
                float(evaluate_at_state(code, state))
                for code in derivative_codes
            ]

        events = []
        event_comparisons = []
        if ode.domain_constraint is not None:
            for comparison in self._simulation_domain_comparisons(ode.domain_constraint):
                if comparison.op not in ('<', '<=', '>', '>='):
                    continue

                def boundary(_time, values, comparison=comparison):
                    state = state_at(values)
                    left = self._compile_simulation_term(comparison.left)
                    right = self._compile_simulation_term(comparison.right)
                    boundary_value = float(
                        evaluate_at_state(left, state)
                        - evaluate_at_state(right, state)
                    )
                    if _time == 0.0 and abs(boundary_value) <= snap_tolerance:
                        crossing_direction = (
                            1 if comparison.op in ('<', '<=') else -1
                        )
                        event_epsilon = max(
                            1e-12, self.ode_solver_tolerance * 1e-3
                        )
                        return -crossing_direction * event_epsilon
                    return boundary_value

                boundary.terminal = True
                boundary.direction = 1 if comparison.op in ('<', '<=') else -1
                events.append(boundary)
                event_comparisons.append(comparison)

        solution = solve_ivp(
            derivatives,
            (0.0, float(self.time_step)),
            initial_values,
            method=self.ode_solver_method,
            max_step=self.ode_solver_step_size,
            rtol=self.ode_solver_tolerance,
            atol=self.ode_solver_tolerance,
            events=events or None,
        )
        if not solution.success:
            raise RuntimeError(f"Hybrid-program ODE integration failed: {solution.message}")

        self.current_state.update(zip(variables, solution.y[:, -1]))
        for comparison, event_times in zip(event_comparisons, solution.t_events):
            if not len(event_times):
                continue
            if isinstance(comparison.left, VarRef):
                self.current_state[comparison.left.name] = self._evaluate_expression(
                    comparison.right, self.current_state
                )
            elif isinstance(comparison.right, VarRef):
                self.current_state[comparison.right.name] = self._evaluate_expression(
                    comparison.left, self.current_state
                )
        self.simulation_elapsed_time += float(solution.t[-1])
        if ode.domain_constraint is not None:
            final_state = dict(self.current_state)
            if not self._evaluate_expression(ode.domain_constraint, final_state):
                raise RuntimeError(
                    "Hybrid-program ODE integration ended outside its evolution domain."
                )
        return True

    def _execute_simulation_program(self, program, probe=False):
        if isinstance(program, HPBlock):
            return self._execute_simulation_program(program.program, probe)
        if isinstance(program, Sequence):
            for child in program.programs:
                if not self._execute_simulation_program(child, probe):
                    return False
            return True
        if isinstance(program, Choice):
            enabled_branches = []
            for branch in self._flatten_simulation_choice(program):
                saved_state = copy.deepcopy(self.current_state)
                try:
                    if self._execute_simulation_program(branch, probe=True):
                        enabled_branches.append(branch)
                finally:
                    self.current_state.clear()
                    self.current_state.update(saved_state)
            if not enabled_branches:
                return False
            selected = int(self.np_random.integers(len(enabled_branches)))
            return self._execute_simulation_program(enabled_branches[selected], probe)
        if isinstance(program, Test):
            return bool(self._evaluate_expression(program.condition, self.current_state))
        if isinstance(program, Assignment):
            value = self._evaluate_expression(program.expr, self.current_state)
            self.current_state[program.var.name] = value
            return True
        if isinstance(program, NondetAssignment):
            raise ValueError(
                "Hybrid-program simulation does not support unbounded nondeterministic assignments."
            )
        if isinstance(program, ODE):
            if probe:
                return (
                    program.domain_constraint is None
                    or bool(self._evaluate_expression(
                        program.domain_constraint, self.current_state
                    ))
                )
            return self._integrate_simulation_ode(program)
        if isinstance(program, Iteration):
            raise ValueError(
                "Nested hybrid-program loops are not supported inside one simulation step."
            )
        if isinstance(program, VarRef):
            raise ValueError(
                f"Unexpanded hybrid-program reference in simulation: {program.name}"
            )
        raise ValueError(
            f"Unsupported hybrid-program node in simulation: {type(program).__name__}"
        )

    def _reset_modal_simulation_state(self):
        constants = {
            constant['name']: self.current_state[constant['name']]
            for constant in self.config['constants']
        }
        max_attempts = self.initial_state_buffer_config.get('max_populate_attempts', 1000)
        for _attempt in range(max_attempts):
            self.current_state.clear()
            self.current_state.update(constants)
            self.current_state.update(self._generate_valid_initial_state(max_attempts=1))
            if self.simulation_initialization_ast is not None:
                if not self._execute_simulation_program(self.simulation_initialization_ast):
                    continue
            if self._simulation_program_is_enabled(self.simulation_step_ast):
                return
        raise RuntimeError(
            "Could not sample an initial state with an enabled execution of the modal program "
            f"after {max_attempts} attempts."
        )

    def _simulation_program_is_enabled(self, program):
        saved_state = copy.deepcopy(self.current_state)
        try:
            return self._execute_simulation_program(program, probe=True)
        finally:
            self.current_state.clear()
            self.current_state.update(saved_state)

    def _step_modal_simulation(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid modal simulation action: {action}")
        original_state = copy.deepcopy(self.current_state)
        control_action = None
        terminated = False
        termination_reason = None
        control_flow_info = None
        if self.control_flow_program_ast is not None:
            control_action = int(np.asarray(action).reshape(-1)[0])
            control_status = self._execute_control_flow(control_action)
            if control_status["terminated"]:
                self.current_state.clear()
                self.current_state.update(original_state)
                terminated = True
                termination_reason = f"control_flow:{control_status['info']}"
                control_flow_info = control_status["info"]
        if not terminated:
            try:
                executed = self._execute_simulation_program(self.simulation_step_ast)
            except Exception:
                self.current_state.clear()
                self.current_state.update(original_state)
                raise
            if not executed:
                self.current_state.clear()
                self.current_state.update(original_state)
                terminated = True
                termination_reason = "no_enabled_hybrid_program_execution"

        reward = sum(
            float(self._evaluate_expression(term, self.current_state))
            for term in self.config.get('simulation_reward_terms', [])
        )
        if control_flow_info is not None:
            reward += float(control_status["reward_delta"])
        truncated = False
        annotation_status = None
        for rule in self.config.get('simulation_predicate_rules', []):
            predicate_holds = bool(
                self._evaluate_expression(rule['predicate'], self.current_state)
            )
            action_key = 'on_success' if predicate_holds else 'on_failure'
            status_key = 'success_status' if predicate_holds else 'failure_status'
            action = rule[action_key]
            status = rule[status_key]
            if action == 'terminate':
                terminated = True
                annotation_status = status if status != 'none' else annotation_status
                termination_reason = f"annotation:{rule['kind']}_{action_key[3:]}"
            elif action == 'truncate':
                truncated = True
                annotation_status = status if status != 'none' else annotation_status
                termination_reason = f"annotation:{rule['kind']}_{action_key[3:]}"

        self.current_step_count += 1
        truncated = truncated or self.current_step_count >= self.max_episode_steps
        observation = self._get_observation()
        info = self._get_info()
        info.update({
            "control_action": control_action,
            "attempted_action": None,
            "action_repaired": False,
            "termination_reason": termination_reason,
            "simulation_mode": True,
            "simulation_elapsed_time": self.simulation_elapsed_time,
        })
        if control_flow_info is not None:
            info["control_flow_info"] = control_flow_info
        if annotation_status is not None:
            info["status"] = annotation_status
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    def _get_ode_system(self, t_ode, y):
        # Map y (from solver) back to named variables
        state_map = {var: y[i] for i, var in enumerate(self.differential_variables)}
        
        # Update current_state for evaluation context, but only for ODE variables
        # The full current_state is updated after the ODE step
        temp_state = copy.copy(self.current_state)
        temp_state.update(state_map)
        
        # Add current ODE solver time 't_ode' to context for ODEs that depend on it
        # This 't_ode' corresponds to the dL variable 't' (time elapsed in current control step)
        temp_state['t'] = t_ode 

        # Helper function to evaluate symbolic function calls within a SymPy expression
        # This is necessary because user-defined functions are Python callables, not SymPy's built-in functions.
        def evaluate_sympy_function_calls_for_ode(sympy_expr_node, current_subs):
            if isinstance(sympy_expr_node, sympy.Function) and str(sympy_expr_node.func) in self.user_defined_functions:
                func_name = str(sympy_expr_node.func)
                evaluated_args = [evaluate_sympy_function_calls_for_ode(arg, current_subs) for arg in sympy_expr_node.args]
                # self.user_defined_functions stores the bound methods, so no need to pass 'self' explicitly.
                return self.user_defined_functions[func_name](*evaluated_args) 
            
            # Recursively apply to arguments of other SymPy expressions
            if hasattr(sympy_expr_node, 'args'): # Check if it's a SymPy expression with arguments
                # Create a new expression with evaluated arguments
                return sympy_expr_node.func(*[evaluate_sympy_function_calls_for_ode(arg, current_subs) for arg in sympy_expr_node.args])
            
            # If it's a simple symbol or literal, substitute its value
            return sympy_expr_node.subs(current_subs)

        # Evaluate differential equations
        derivs = []
        for var_name in self.differential_variables:
            expr = self.differential_equations[var_name] # This is already a SymPy expression
            
            # Create a substitution dictionary from temp_state and sympy_symbols
            subs_dict = {self.sympy_symbols[k]: v for k, v in temp_state.items() if k in self.sympy_symbols}
            
            # Evaluate the ODE expression, handling symbolic function calls
            evaluated_ode_expr = self._evaluate_sympy_function_calls(expr, subs_dict)
            
            # The result should now be a numerical value or a simple SymPy expression
            # that can be converted to float.
            derivs.append(float(evaluated_ode_expr))
        return derivs

    def _run_continuous_dynamics(self, action_to_apply):
        self.last_dynamics_failure_reason = None
        # Apply action to relevant variables before ODE integration
        # This block is now only relevant if action_space_type is 'continuous'
        # and controlFunc is NOT being used to set variables.
        # If controlFunc is active, the variables like 'ar' are already set in self.current_state.
        if self.action_space_type == 'continuous' and self.action_variable_names:
            for i, var_name in enumerate(self.action_variable_names):
                self.current_state[var_name] = action_to_apply[i]

        # The dL model resets its local clock before each control-cycle flow.
        self.current_state['t'] = 0.0
        
        # Update dL 'T' (global time remaining) is handled by ODE `T' = -1`

        # Prepare initial state for ODE solver
        y0 = [self.current_state[var] for var in self.differential_variables]
        
        t_span = (0, self.time_step) # Integrate over the control time step 'ep'

        ode_sympy_locals = {k: v for k, v in self.sympy_symbols.items()}
        ode_sympy_locals.update({
            'sympy': sympy,
            'And': sympy.And,
            'Le': sympy.Le,
            'Ge': sympy.Ge,
            'Lt': sympy.Lt,
            'Gt': sympy.Gt,
            'Float': sympy.Float,
        })
        for func_name in self.user_defined_functions:
            ode_sympy_locals[func_name] = sympy.Function(func_name)

        constraint_sympy_expr = None
        domain_comparisons = None
        if self.differential_domain_constraint:
            constraint_sympy_expr = sympy.sympify(
                self.differential_domain_constraint,
                locals=ode_sympy_locals,
            )

            def extract_domain_comparisons(expression):
                if isinstance(expression, sympy.core.relational.Relational):
                    if expression.rel_op in ('<', '<=', '>', '>='):
                        return [expression]
                    return None
                if expression is sympy.true:
                    return []
                if isinstance(expression, sympy.And):
                    comparisons = []
                    for child in expression.args:
                        child_comparisons = extract_domain_comparisons(child)
                        if child_comparisons is None:
                            return None
                        comparisons.extend(child_comparisons)
                    return comparisons
                return None

            domain_comparisons = extract_domain_comparisons(
                constraint_sympy_expr
            )
            initial_subs = {
                self.sympy_symbols[name]: value
                for name, value in self.current_state.items()
                if name in self.sympy_symbols
            }
            if not bool(self._evaluate_sympy_function_calls(
                constraint_sympy_expr, initial_subs
            )):
                print(
                    "Differential domain constraint is false at the start "
                    f"of integration: {self.differential_domain_constraint}"
                )
                self.last_dynamics_failure_reason = "ode_domain_violation"
                return True

        events = []
        event_comparisons = []
        if domain_comparisons is not None:
            snap_tolerance = max(1e-9, self.ode_solver_tolerance * 10)
            event_epsilon = max(1e-12, self.ode_solver_tolerance * 1e-3)
            for comparison in domain_comparisons:
                exits_toward_positive = comparison.rel_op in ('<', '<=')
                inside_sign = -1 if exits_toward_positive else 1
                initial_boundary_value = float(
                    self._evaluate_sympy_function_calls(
                        comparison.lhs, initial_subs
                    )
                    - self._evaluate_sympy_function_calls(
                        comparison.rhs, initial_subs
                    )
                )
                initially_on_boundary = (
                    abs(initial_boundary_value) <= snap_tolerance
                )

                def boundary(
                    _time,
                    values,
                    comparison=comparison,
                    inside_sign=inside_sign,
                    initially_on_boundary=initially_on_boundary,
                ):
                    state = dict(self.current_state)
                    state.update(zip(self.differential_variables, values))
                    substitutions = {
                        self.sympy_symbols[name]: value
                        for name, value in state.items()
                        if name in self.sympy_symbols
                    }
                    boundary_value = float(
                        self._evaluate_sympy_function_calls(
                            comparison.lhs, substitutions
                        )
                        - self._evaluate_sympy_function_calls(
                            comparison.rhs, substitutions
                        )
                    )
                    if (
                        initially_on_boundary
                        and abs(boundary_value) <= event_epsilon
                    ):
                        return inside_sign * event_epsilon
                    return boundary_value

                boundary.terminal = True
                boundary.direction = 1 if exits_toward_positive else -1
                events.append(boundary)
                event_comparisons.append(comparison)
        elif constraint_sympy_expr is not None:
            self._debug_print(
                "Evolution-domain event stopping only supports conjunctions "
                "of inequalities; the final-state domain check remains active."
            )

        sol = solve_ivp(
            self._get_ode_system,
            t_span,
            y0,
            method=self.ode_solver_method,
            max_step=self.ode_solver_step_size,
            rtol=self.ode_solver_tolerance,
            atol=self.ode_solver_tolerance,
            events=events or None,
        )

        if sol.status >= 0: # Success or a terminal evolution-domain event
            # Update state with the last point from the solver
            for i, var_name in enumerate(self.differential_variables):
                self.current_state[var_name] = sol.y[i, -1]

            for comparison, event_times in zip(event_comparisons, sol.t_events):
                if not len(event_times):
                    continue
                substitutions = {
                    self.sympy_symbols[name]: value
                    for name, value in self.current_state.items()
                    if name in self.sympy_symbols
                }
                if (
                    isinstance(comparison.lhs, sympy.Symbol)
                    and str(comparison.lhs) in self.differential_variables
                ):
                    self.current_state[str(comparison.lhs)] = float(
                        self._evaluate_sympy_function_calls(
                            comparison.rhs, substitutions
                        )
                    )
                elif (
                    isinstance(comparison.rhs, sympy.Symbol)
                    and str(comparison.rhs) in self.differential_variables
                ):
                    self.current_state[str(comparison.rhs)] = float(
                        self._evaluate_sympy_function_calls(
                            comparison.lhs, substitutions
                        )
                    )

            # The solver time is the elapsed dL time, including early domain stops.
            self.current_state['t'] = sol.t[-1]

            if constraint_sympy_expr is not None:
                concrete_subs = {
                    self.sympy_symbols[name]: value
                    for name, value in self.current_state.items()
                    if name in self.sympy_symbols
                }
                constraint_met = self._evaluate_sympy_function_calls(
                    constraint_sympy_expr, concrete_subs
                )

                if not bool(constraint_met): # Convert SymPy boolean to Python bool
                    print(f"Differential domain constraint violated: {self.differential_domain_constraint}")
                    self._debug_print(f"DEBUG: _run_continuous_dynamics returning True due to domain violation.")
                    self.last_dynamics_failure_reason = "ode_domain_violation"
                    return True # Indicate episode should terminate due to violation
            self._debug_print(f"DEBUG: _run_continuous_dynamics returning False (solver success, no domain violation).")
            return False # Indicate episode continues
        else:
            print(f"ODE solver failed: {sol.message}")
            self._debug_print(f"DEBUG: _run_continuous_dynamics returning True due to solver failure.")
            self.last_dynamics_failure_reason = "ode_solver_failure"
            return True # Indicate episode should terminate

    def action_masks(self):
        """Boolean mask of controller choice paths whose guards hold in the current state."""
        if self.action_space_type != 'discrete' or not self.control_flow_program_ast:
            return np.ones(getattr(self.action_space, 'n', 1), dtype=bool)
        saved_state = dict(self.current_state)
        mask = np.zeros(len(self.control_choice_paths), dtype=bool)
        try:
            for index, path in enumerate(self.control_choice_paths):
                self.current_state.clear()
                self.current_state.update(saved_state)
                result = self._execute_hp_node(self.control_flow_program_ast, path, [0])
                mask[index] = not result["terminated"]
        finally:
            self.current_state.clear()
            self.current_state.update(saved_state)
        if not mask.any():
            # No legal choice: let the environment report the failed guard instead of hiding it.
            mask[:] = True
        return mask

    def _execute_control_flow(self, action_choice: int) -> Dict[str, Any]:
        """
        Executes the discrete control flow logic defined by the controlFunc AST.
        Modifies self.current_state based on assignments and checks conditions.
        Returns a dictionary with execution status (e.g., 'terminated', 'reward_delta').
        """
        status = {"terminated": False, "reward_delta": 0.0, "info": ""}

        if not self.control_flow_program_ast:
            self._debug_print("DEBUG: _execute_control_flow called but no control_flow_program_ast defined.")
            status["info"] = "no_control_flow_program"
            return status

        if not self.action_space.contains(action_choice):
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "invalid_control_choice"
            return status

        try:
            path_index = int(action_choice)
        except (TypeError, ValueError, OverflowError):
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "invalid_control_choice"
            return status

        if path_index < 0 or path_index >= len(self.control_choice_paths):
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "invalid_control_choice"
            return status

        self._debug_print(f"DEBUG: Executing complete control path {path_index}.")
        cursor = [0]
        return self._execute_hp_node(
            self.control_flow_program_ast,
            self.control_choice_paths[path_index],
            cursor,
        )

    def _execute_hp_node(
        self,
        hp_node: HybridProgram,
        choice_path=(),
        choice_cursor=None,
    ) -> Dict[str, Any]:
        """
        Recursively executes a hybrid program node.
        Returns execution status, potentially modifying self.current_state.
        """
        status = {"terminated": False, "reward_delta": 0.0, "info": ""}
        if choice_cursor is None:
            choice_cursor = [0]

        if isinstance(hp_node, HPBlock):
            # Execute the program inside the block
            return self._execute_hp_node(hp_node.program, choice_path, choice_cursor)
        
        elif isinstance(hp_node, Sequence):
            for program in hp_node.programs:
                sub_status = self._execute_hp_node(program, choice_path, choice_cursor)
                status["terminated"] = status["terminated"] or sub_status["terminated"]
                status["reward_delta"] += sub_status["reward_delta"]
                if status["terminated"]: # If any part of sequence terminates, the whole sequence terminates
                    status["info"] = sub_status["info"] or "sequence_terminated"
                    break
            return status

        elif isinstance(hp_node, Choice):
            if choice_cursor[0] >= len(choice_path):
                status.update(
                    terminated=True,
                    reward_delta=-100.0,
                    info="invalid_control_choice_path",
                )
                return status
            branch = hp_node.left if choice_path[choice_cursor[0]] == 0 else hp_node.right
            choice_cursor[0] += 1
            return self._execute_hp_node(branch, choice_path, choice_cursor)
        
        elif isinstance(hp_node, Test):
            condition_met = self._evaluate_expression(hp_node.condition, self.current_state)
            if not condition_met:
                self._debug_print(f"DEBUG: Test condition '{hp_node.condition._repr_indented(0)}' failed. Terminating.")
                status["terminated"] = True
                status["reward_delta"] = -100.0 # Penalty for failing a test
                status["info"] = "test_condition_failed"
            return status

        elif isinstance(hp_node, Assignment):
            var_name = hp_node.var.name
            expr_value = self._evaluate_expression(hp_node.expr, self.current_state)
            handler = getattr(self, '_assignment_handler', None)
            if handler is not None:
                expr_value = handler(var_name, expr_value)
            self.current_state[var_name] = expr_value
            self._debug_print(f"DEBUG: Assignment: {var_name} := {expr_value}")
            return status

        elif isinstance(hp_node, NondetAssignment):
            handler = getattr(self, '_nondeterministic_assignment_handler', None)
            if handler is not None:
                self.current_state[hp_node.var.name] = float(handler(hp_node.var.name))
                return status
            # This is more complex. For now, we'll raise an error or apply a default.
            # A proper implementation would sample from the allowed range.
            self._debug_print(f"ERROR: Nondeterministic assignment '{hp_node.var.name} := *' not yet fully supported. Defaulting to 0.0.")
            self.current_state[hp_node.var.name] = 0.0 # Default fallback
            status["info"] = "nondet_assignment_defaulted"
            return status

        elif isinstance(hp_node, Iteration):
            self._debug_print("WARNING: Iteration node encountered in controlFunc. Iterations are typically handled by the RL loop, not directly executed here.")
            # If an iteration is part of controlFunc, it implies the agent decides to loop.
            # This is usually a higher-level concept. For now, treat as a single execution.
            return self._execute_hp_node(hp_node.program, choice_path, choice_cursor)

        elif isinstance(hp_node, VarRef):
            # If the HP is just a reference to another HPDef, resolve and execute it.
            # We need to look up the HPDef by name.
            # This requires a map of HPDef names to their bodies.
            # For simplicity, we'll assume the generator only passes the 'controlFunc' AST directly.
            # If other HP references are expected in controlFunc, the environment needs a way to resolve them.
            self._debug_print(f"ERROR: HP VarRef '{hp_node.name}' encountered in control flow. Direct HP references within controlFunc are not yet supported for recursive execution.")
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "unsupported_hp_varref_in_control_flow"
            return status
        
        elif isinstance(hp_node, ODE):
            self._debug_print("WARNING: ODE node encountered in controlFunc. ODEs are handled by _run_continuous_dynamics, not discrete control flow.")
            # This indicates a malformed controlFunc if it directly contains an ODE.
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "unexpected_ode_in_control_flow"
            return status

        else:
            self._debug_print(f"ERROR: Unsupported HybridProgram node type in _execute_hp_node: {type(hp_node)}")
            status["terminated"] = True
            status["reward_delta"] = -100.0
            status["info"] = "unsupported_hp_node_type"
            return status


    def step(self, action):
        if self.simulation_mode:
            return self._step_modal_simulation(action)

        terminated = False
        truncated = False
        reward = 0.0 # Initialize reward
        termination_reason = None
        control_action = None
        attempted_action = None
        action_repaired = False

        # Execute discrete control flow based on agent's action
        if self.control_flow_program_ast:
            action_values = np.asarray(action).reshape(-1)
            control_action = int(action_values[0])
            invalid_mode = self.config.get('invalid_action_mode', 'mask')
            if invalid_mode not in ('mask', 'penalize', 'repair'):
                raise ValueError(f"Unsupported invalid_action_mode: {invalid_mode}")
            if invalid_mode == 'repair':
                legal = self.action_masks()
                in_range = 0 <= control_action < len(legal)
                if in_range and not legal[control_action] and legal.any() and not legal.all():
                    attempted_action = control_action
                    control_action = int(np.flatnonzero(legal)[0])
                    action = control_action
                    action_repaired = True
                    reward += self.config.get('repair_penalty', 0.0)
            # 'mask' relies on a mask-aware algorithm; an illegal action that still arrives is penalized.
            control_flow_status = self._execute_control_flow(action)
            terminated = control_flow_status["terminated"]
            reward += control_flow_status["reward_delta"]
            if terminated:
                termination_reason = f"control_flow:{control_flow_status['info']}"
                self._debug_print(f"DEBUG: step method: Terminated by control flow: {control_flow_status['info']}")
                observation = self._get_observation()
                info = self._get_info()
                info["control_flow_info"] = control_flow_status["info"]
                info["control_action"] = control_action
                info["attempted_action"] = attempted_action
                info["action_repaired"] = action_repaired
                info["termination_reason"] = termination_reason
                if self.render_mode == "human":
                    self._render_frame()
                return observation, reward, terminated, truncated, info
        else:
            # If no controlFunc, then the action is directly applied to continuous variables
            # This path is for environments without a controlFunc or where it's ignored
            # and action is directly continuous values for ODE variables.
            if self.action_space_type == 'continuous':
                action_to_apply = action
            else:
                # This case should ideally not be reached if action_space_type is discrete
                # but control_flow_program_ast is None.
                # If it is reached, it means a discrete action was given without a controlFunc to interpret it.
                self._debug_print(f"WARNING: Discrete action received but no controlFunc to interpret it. Action: {action}")
                action_to_apply = [] # No direct action for ODEs
        
        # Run continuous dynamics (ODE integration)
        # If controlFunc was active, it would have set the control variables in self.current_state.
        # So, _run_continuous_dynamics will pick them up. action_to_apply will be empty in this case.
        solver_failed = self._run_continuous_dynamics(action if self.action_space_type == 'continuous' else [])
        self._debug_print(f"DEBUG: step method: solver_failed = {solver_failed}")
        if solver_failed:
            terminated = True # Terminate if ODE solver fails or domain constraint violated
            termination_reason = self.last_dynamics_failure_reason or "ode_failure"
            reward += self.config.get('ode_failure_penalty', -100.0) # Penalty for ODE failure (configurable)

        # Evaluate reward (from problem definition)
        reward_expr = self.config.get('reward_function_body', '0.0')
        self._debug_print(f"DEBUG: step method: reward_expr before evaluation: '{reward_expr}' (type: {type(reward_expr)})")
        reward += self._evaluate_expression(reward_expr, self.current_state) # Add to existing reward

        # Evaluate termination condition (from problem definition)
        termination_expr = self.config.get('termination_condition_body', 'False')
        self._debug_print(f"DEBUG: step method: termination_expr for evaluation: '{termination_expr}'")
        if not terminated: # Only evaluate if not already terminated by solver failure or control flow
            evaluated_termination_cond = self._evaluate_expression(termination_expr, self.current_state)
            self._debug_print(f"DEBUG: step method: evaluated_termination_cond = {evaluated_termination_cond}")
            terminated = bool(evaluated_termination_cond)
            if terminated:
                if not self._evaluate_expression(
                    self.config.get('problem', {}).get('goal', 'True'),
                    self.current_state
                ):
                    termination_reason = "model_safety_postcondition_violation"
                elif any(
                    not self._evaluate_expression(invariant, self.current_state)
                    for invariant in self.config.get('problem', {}).get('loop_invariants', [])
                ):
                    termination_reason = "model_loop_invariant_violation"
                else:
                    termination_reason = "model_termination_condition"
        self._debug_print(f"DEBUG: step method: final terminated status = {terminated}")

        # Check for truncation (max steps or global time)
        self.current_step_count += 1
        if self.current_step_count >= self.max_episode_steps:
            truncated = True
            if not terminated:
                termination_reason = "time_limit_truncation"
        
        # Update global_time_remaining from current_state['T'] after ODE step.
        self.global_time_remaining = self.current_state.get('T', 0.0)

        # Get observation
        observation = self._get_observation()

        # Render (if applicable)
        if self.render_mode == "human":
            self._render_frame()

        info = self._get_info()
        info["control_action"] = control_action
        info["attempted_action"] = attempted_action
        info["action_repaired"] = action_repaired
        info["termination_reason"] = termination_reason
        info["ode_failure_reason"] = self.last_dynamics_failure_reason
        return observation, reward, terminated, truncated, info

    def _generate_valid_initial_state(self, max_attempts=None):
        # Generates a single initial state that satisfies the initial_condition.
        # Returns a dictionary of sampled variable values, 't', and 'T'.
        # Raises RuntimeError if a valid state cannot be found within max_attempts.
        if max_attempts is None:
            max_attempts = self.initial_state_buffer_config.get('max_populate_attempts', 1000)

        initial_cond_expr = self.config['problem']['initial_condition']
        
        for attempt in range(max_attempts):
            sampled_vars = {}
            # Sample values for variables
            for var_config in self.config['variables']:
                var_name = var_config['name']
                if var_name in self.initial_state_samplers:
                    sampled_vars[var_name] = self.initial_state_samplers[var_name]()
                else:
                    sampled_vars[var_name] = 0.0 # Default if no sampler

            # Prepare a full state dictionary for evaluation (constants + sampled vars + t, T)
            # Start with current constants (which are in self.current_state after _setup_constants)
            temp_eval_state = copy.copy(self.current_state) 
            temp_eval_state.update(sampled_vars) # Overlay sampled variables

            # Set 't' for initial condition evaluation
            temp_eval_state['t'] = 0.0
            # Ensure 'T' is present for evaluation. If it's a variable, its sampled value is used.
            # If it's a constant, its value is already in temp_eval_state.
            # If it's neither, provide a default.
            if 'T' not in temp_eval_state:
                temp_eval_state['T'] = 10.0 # Default global time remaining

            # Evaluate initial_condition using this temporary state
            initial_cond_met = self._evaluate_expression(initial_cond_expr, temp_eval_state)
            
            if initial_cond_met:
                # Return only the parts that need to be set into self.current_state
                # after constants are already loaded.
                result_state_update = {k: v for k, v in sampled_vars.items()}
                result_state_update['t'] = 0.0
                result_state_update['T'] = temp_eval_state['T'] # Use the T that was used for evaluation
                return result_state_update

        raise RuntimeError(f"Failed to find a valid initial state after {max_attempts} attempts "
                           f"for initial condition: '{initial_cond_expr}'")

    def _build_initial_state_sample_ranges(self):
        sample_ranges = {}
        for var_name, sampling_config in self.config.get('initial_state_sampling_config', {}).items():
            if 'value' in sampling_config:
                value = self._evaluate_expression(sampling_config['value'], {})
                sample_ranges[var_name] = (float(value), float(value))
            elif 'min' in sampling_config and 'max' in sampling_config:
                low = self._evaluate_expression(sampling_config['min'], {})
                high = self._evaluate_expression(sampling_config['max'], {})
                sample_ranges[var_name] = (float(low), float(high))
            else:
                sample_ranges[var_name] = (-100.0, 100.0)
        return sample_ranges

    def _expr_to_z3(self, expr: str, z3_symbols: Dict[str, z3.ArithRef], fixed_constants_map: Dict[str, float]):
        expr = expr.strip()
        if not expr:
            return z3.BoolVal(True)

        def find_function_body(function_name: str):
            for func_config in self.config.get('function_definitions', []):
                if func_config.get('name') == function_name:
                    return func_config
            return None

        def is_boolean_function_definition(function_name: str):
            func_def = find_function_body(function_name)
            if func_def is None:
                return False
            body = func_def.get('body')
            if not isinstance(body, str):
                return False
            try:
                parsed = ast.parse(body, mode='eval').body
            except SyntaxError:
                return False
            if isinstance(parsed, ast.BoolOp):
                return True
            if isinstance(parsed, ast.Compare):
                return True
            if isinstance(parsed, ast.UnaryOp) and isinstance(parsed.op, ast.Not):
                return True
            if isinstance(parsed, ast.Call):
                func = parsed.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'self':
                    return is_boolean_function_definition(func.attr.replace('_impl', '').lstrip('_'))
                if isinstance(func, ast.Name):
                    return is_boolean_function_definition(func.id)
            if isinstance(parsed, ast.Name):
                return parsed.id in ('True', 'False')
            return False

        def convert(node, local_bindings=None):
            local_bindings = {} if local_bindings is None else local_bindings

            if isinstance(node, ast.Expression):
                return convert(node.body, local_bindings)
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return z3.And(*[convert(v, local_bindings) for v in node.values])
                if isinstance(node.op, ast.Or):
                    return z3.Or(*[convert(v, local_bindings) for v in node.values])
            if isinstance(node, ast.UnaryOp):
                operand = convert(node.operand, local_bindings)
                if isinstance(node.op, ast.Not):
                    return z3.Not(operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return operand
            if isinstance(node, ast.BinOp):
                left = convert(node.left, local_bindings)
                right = convert(node.right, local_bindings)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                if isinstance(node.op, ast.Mod):
                    return left % right
                raise ValueError(f"Unsupported binary operator in initial condition: {type(node.op).__name__}")
            if isinstance(node, ast.Compare):
                left = convert(node.left, local_bindings)
                comparisons = []
                for op, comp in zip(node.ops, node.comparators):
                    right = convert(comp, local_bindings)
                    if isinstance(op, ast.Gt):
                        comparisons.append(left > right)
                    elif isinstance(op, ast.GtE):
                        comparisons.append(left >= right)
                    elif isinstance(op, ast.Lt):
                        comparisons.append(left < right)
                    elif isinstance(op, ast.LtE):
                        comparisons.append(left <= right)
                    elif isinstance(op, ast.Eq):
                        comparisons.append(left == right)
                    elif isinstance(op, ast.NotEq):
                        comparisons.append(left != right)
                    else:
                        raise ValueError(f"Unsupported comparison operator in initial condition: {type(op).__name__}")
                    left = right
                if not comparisons:
                    return z3.BoolVal(True)
                if len(comparisons) == 1:
                    return comparisons[0]
                return z3.And(*comparisons)
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'self':
                    attr_name = node.attr
                    if attr_name.endswith('_impl'):
                        attr_name = attr_name[:-5]
                    attr_name = attr_name.lstrip('_')
                    if attr_name in local_bindings:
                        return local_bindings[attr_name]
                    if attr_name in z3_symbols:
                        return z3_symbols[attr_name]
                    if attr_name in fixed_constants_map:
                        return z3.RealVal(str(fixed_constants_map[attr_name]))
                    if attr_name in ('True', 'False'):
                        return z3.BoolVal(attr_name == 'True')
                    return z3.Real(attr_name)
                raise ValueError(f"Unsupported attribute access in initial condition: {ast.dump(node)}")
            if isinstance(node, ast.Name):
                if node.id in local_bindings:
                    return local_bindings[node.id]
                if node.id in z3_symbols:
                    return z3_symbols[node.id]
                if node.id in fixed_constants_map:
                    return z3.RealVal(str(fixed_constants_map[node.id]))
                if node.id in ('True', 'False'):
                    return z3.BoolVal(node.id == 'True')
                return z3.Real(node.id)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool):
                    return z3.BoolVal(node.value)
                if isinstance(node.value, (int, float)):
                    return z3.RealVal(str(node.value))
                return z3.RealVal(str(node.value))
            if isinstance(node, ast.Num):
                return z3.RealVal(str(node.n))
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                    method_name = node.func.attr
                    if method_name.endswith('_impl'):
                        method_name = method_name[:-5]
                    method_name = method_name.lstrip('_')
                    func_def = find_function_body(method_name)
                    if func_def is not None:
                        arg_names = func_def.get('args', [])
                        if not node.args and not arg_names:
                            return convert(ast.parse(func_def['body'], mode='eval').body, local_bindings)
                        new_local_bindings = dict(local_bindings)
                        for arg_name, arg_node in zip(arg_names, node.args):
                            new_local_bindings[arg_name] = convert(arg_node, local_bindings)
                        return convert(ast.parse(func_def['body'], mode='eval').body, new_local_bindings)

                    if not node.args:
                        if method_name in local_bindings:
                            return local_bindings[method_name]
                        if method_name in z3_symbols:
                            return z3_symbols[method_name]
                        if method_name in fixed_constants_map:
                            return z3.RealVal(str(fixed_constants_map[method_name]))
                        if is_boolean_function_definition(method_name):
                            return z3.Bool(method_name)
                        return z3.Real(method_name)

                    if method_name in ('min', 'max'):
                        values = [convert(arg, local_bindings) for arg in node.args]
                        if len(values) == 2:
                            a, b = values
                            return z3.If(a <= b, a, b) if method_name == 'min' else z3.If(a >= b, a, b)
                        return values[0]

                    raise ValueError(f"Unsupported method call in initial condition: {method_name}")

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    func_def = find_function_body(func_name)
                    if func_def is not None:
                        arg_names = func_def.get('args', [])
                        if not node.args and not arg_names:
                            return convert(ast.parse(func_def['body'], mode='eval').body, local_bindings)
                        new_local_bindings = dict(local_bindings)
                        for arg_name, arg_node in zip(arg_names, node.args):
                            new_local_bindings[arg_name] = convert(arg_node, local_bindings)
                        return convert(ast.parse(func_def['body'], mode='eval').body, new_local_bindings)

                    if func_name in ('min', 'max'):
                        values = [convert(arg, local_bindings) for arg in node.args]
                        if len(values) == 2:
                            a, b = values
                            return z3.If(a <= b, a, b) if func_name == 'min' else z3.If(a >= b, a, b)
                        if values:
                            return values[0]
                    if func_name == 'pow' and len(node.args) == 2:
                        base, exp = [convert(arg, local_bindings) for arg in node.args]
                        return base ** exp
                    if func_name in ('float', 'int', 'bool') and len(node.args) == 1:
                        return convert(node.args[0], local_bindings)
                    # Treat unknown function calls as a symbolic placeholder for the function name.
                    if not node.args:
                        if is_boolean_function_definition(func_name):
                            return z3.Bool(func_name)
                        return z3.Real(func_name)
                    return z3.Bool(func_name)

                raise ValueError(f"Unsupported call in initial condition: {ast.dump(node)}")
            raise ValueError(f"Unsupported expression node in initial condition: {type(node).__name__}")

        try:
            parsed = ast.parse(expr, mode='eval')
            return convert(parsed)
        except Exception as exc:
            raise ValueError(f"Failed to translate initial condition to Z3: {expr} -> {exc}") from exc

    def _populate_initial_state_buffer_diverse_z3(self):
        # Populate a diverse set of valid initial states using Z3 optimizer.
        capacity = self.initial_state_buffer_config.get('capacity', 100)
        max_populate_attempts_total = self.initial_state_buffer_config.get('max_populate_attempts', 1000)
        self._debug_print(f"DEBUG: Populating diverse Z3 initial state buffer (capacity: {capacity}, max_attempts_total: {max_populate_attempts_total})")

        sample_ranges = self._build_initial_state_sample_ranges()
        z3_symbols = {}
        fixed_constants_map = {}

        for var_config in self.config.get('variables', []):
            var_name = var_config['name']
            z3_symbols[var_name] = z3.Real(var_name)
        for const_config in self.config.get('constants', []):
            const_name = const_config['name']
            const_value = const_config.get('value')
            if const_value is not None and isinstance(const_value, (int, float)):
                fixed_constants_map[const_name] = float(const_value)
            elif const_value is not None:
                try:
                    numeric_value = float(const_value)
                    fixed_constants_map[const_name] = numeric_value
                except (TypeError, ValueError):
                    z3_symbols[const_name] = z3.Real(const_name)

        opt = z3.Optimize()
        opt.set(timeout=10000)

        initial_condition_expr = self.config.get('problem', {}).get('initial_condition', 'True')
        try:
            z3_initial_condition = self._expr_to_z3(initial_condition_expr, z3_symbols, fixed_constants_map)
            opt.add(z3_initial_condition)
        except ValueError as e:
            print(f"CRITICAL ERROR: Failed to convert initial condition to Z3 during CISB population: {e}")
            return

        for var_name, z3_var in z3_symbols.items():
            low, high = sample_ranges.get(var_name, (-100.0, 100.0))
            opt.add(z3_var >= low, z3_var <= high)

        progress_width = 30
        progress_is_tty = sys.stderr.isatty()
        progress_interval = max(1, min(100, max_populate_attempts_total // 20))
        last_progress_attempt = -1

        def report_progress(final=False):
            nonlocal last_progress_attempt
            filled = min(
                progress_width,
                progress_width * len(self.initial_state_buffer) // max(capacity, 1),
            )
            line = (
                "\rZ3 initial-state buffer: [%s%s] %d/%d states "
                "| attempts %d/%d"
                % (
                    "#" * filled,
                    "-" * (progress_width - filled),
                    len(self.initial_state_buffer),
                    capacity,
                    attempts_made,
                    max_populate_attempts_total,
                )
            )
            if progress_is_tty:
                sys.stderr.write(line)
                if final:
                    sys.stderr.write("\n")
                sys.stderr.flush()
                last_progress_attempt = attempts_made
            elif final or attempts_made % progress_interval == 0:
                print(line.lstrip("\r"), file=sys.stderr, flush=True)
                last_progress_attempt = attempts_made

        attempts_made = 0
        report_progress()
        while len(self.initial_state_buffer) < capacity and attempts_made < max_populate_attempts_total:
            attempts_made += 1
            opt.push()

            soft_constraints_added = False
            for var_name, z3_var in z3_symbols.items():
                if var_name in fixed_constants_map:
                    continue
                low, high = sample_ranges.get(var_name, (-100.0, 100.0))
                if low == high:
                    continue
                target = self.np_random.uniform(low, high)
                opt.add_soft(z3_var == target, weight=1)
                soft_constraints_added = True

            check_result = opt.check()
            if check_result == z3.sat:
                model = opt.model()
                solved_values = {}
                for k, v in z3_symbols.items():
                    evaluated_val = model.evaluate(v, model_completion=True)
                    if evaluated_val is not None:
                        if hasattr(evaluated_val, 'as_fraction'):
                            solved_values[k] = float(evaluated_val.as_fraction())
                        elif hasattr(evaluated_val, 'as_long'):
                            solved_values[k] = float(evaluated_val.as_long())
                        else:
                            solved_values[k] = float(str(evaluated_val))
                solved_values.update(fixed_constants_map)

                if 't' not in solved_values:
                    solved_values['t'] = 0.0
                if 'T' not in solved_values:
                    solved_values['T'] = 10.0

                full_state_to_store = OrderedDict()
                full_state_to_store.update(self.current_state)
                full_state_to_store.update(solved_values)
                self.initial_state_buffer.append(copy.deepcopy(full_state_to_store))
                self._debug_print(f"DEBUG: Added diverse Z3-generated state to buffer. Buffer size: {len(self.initial_state_buffer)}")

                blocking_conditions = []
                for var_name, z3_var in z3_symbols.items():
                    val = solved_values.get(var_name)
                    if val is not None:
                        diversity_epsilon = 0.1
                        blocking_conditions.append(z3.Or(z3_var < val - diversity_epsilon, z3_var > val + diversity_epsilon))

                if blocking_conditions:
                    opt.add(z3.Or(blocking_conditions))
                opt.pop()
            elif check_result == z3.unsat:
                print(f"DEBUG: Z3 UNSAT during diverse buffer population. No more states available.")
                opt.pop()
                break
            else:
                print(f"DEBUG: Z3 UNKNOWN during diverse buffer population. Retrying with new soft constraints.")
                opt.pop()
            report_progress()

        if progress_is_tty:
            sys.stderr.write("\n")
            sys.stderr.flush()
        elif last_progress_attempt != attempts_made:
            report_progress(final=True)
        if len(self.initial_state_buffer) < capacity:
            print(f"Warning: Diverse initial state buffer could only be filled to {len(self.initial_state_buffer)}/{capacity} after {attempts_made} attempts.")

    def _populate_initial_state_buffer(self):
        # Populates the initial state buffer with valid initial states.
        isb_config = self.initial_state_buffer_config
        strategy = isb_config.get('strategy', 'random')
        if strategy == 'z3_diverse':
            self._populate_initial_state_buffer_diverse_z3()
            return

        capacity = isb_config.get('capacity', 100)
        max_populate_attempts_total = isb_config.get('max_populate_attempts', 1000) # Total attempts for population

        self._debug_print(f"DEBUG: Populating initial state buffer (capacity: {capacity}, max_attempts_total: {max_populate_attempts_total})")

        attempts_made = 0
        while len(self.initial_state_buffer) < capacity and attempts_made < max_populate_attempts_total:
            try:
                valid_state_update = self._generate_valid_initial_state(max_attempts=100)
                full_state_to_store = OrderedDict()
                full_state_to_store.update(self.current_state)
                full_state_to_store.update(valid_state_update)
                self.initial_state_buffer.append(full_state_to_store)
                self._debug_print(f"DEBUG: Added state to buffer. Buffer size: {len(self.initial_state_buffer)}")
            except RuntimeError as e:
                self._debug_print(f"DEBUG: Failed to generate a valid state during population: {e}. Continuing attempts.")
            attempts_made += 1
        
        if len(self.initial_state_buffer) < capacity:
            print(f"Warning: Initial state buffer could only be filled to {len(self.initial_state_buffer)}/{capacity} after {attempts_made} attempts.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.simulation_elapsed_time = 0.0
        self.current_state = OrderedDict()

        # Re-apply constants (already done in __init__, but good to ensure for reset)
        self._setup_constants()

        # Ensure functions are set up for _evaluate_expression calls during reset
        self._setup_functions()

        isb_enabled = self.initial_state_buffer_config.get('enabled', False)
        
        if isb_enabled and len(self.initial_state_buffer) > 0:
            # Use a state from the buffer
            self.current_state = self.initial_state_buffer.popleft()
            self._debug_print(f"DEBUG: Resetting from initial state buffer. Buffer size: {len(self.initial_state_buffer)}")
        else:
            # Generate a new initial state
            try:
                # This will return a dict of sampled variables, 't', and 'T'
                new_state_update = self._generate_valid_initial_state()
                self.current_state.update(new_state_update) # Merge sampled vars, t, T into current_state (which has constants)
                self._debug_print(f"DEBUG: Generated new initial state.")

                # If ISB is enabled, add this newly generated state to the buffer
                if isb_enabled:
                    # Store the full current_state
                    self.initial_state_buffer.append(copy.deepcopy(self.current_state))
                    self._debug_print(f"DEBUG: Added newly generated state to buffer. Buffer size: {len(self.initial_state_buffer)}")

            except RuntimeError as e:
                print(f"Error during reset: {e}. Using default initial state (all zeros for variables).")
                # Fallback: if generation fails, initialize variables to 0.0
                for var_config in self.config['variables']:
                    var_name = var_config['name']
                    if var_name not in self.current_state: # Don't overwrite constants
                        self.current_state[var_name] = 0.0
                self.current_state['t'] = 0.0
                self.current_state['T'] = 10.0 # Default if not set

        # Ensure 'ep' (time_step) is correctly set from constants if available
        ep_value = None
        ep_const_config = next((c for c in self.config['constants'] if c['name'] == 'ep'), None)
        if ep_const_config:
            ep_value = self.current_state['ep'] # Value already evaluated and stored in current_state
        else: # Check if 'ep' is a function without arguments
            ep_func_config = next((f for f in self.config['function_definitions'] if f['name'] == 'ep' and not f['args']), None)
            if ep_func_config:
                ep_value = self.user_defined_functions['ep']() # Call the generated _ep_impl method

        if ep_value is not None:
            self.time_step = ep_value

        if self.simulation_mode:
            self._reset_modal_simulation_state()
        
        # Ensure global_time_remaining is updated from the state
        self.global_time_remaining = self.current_state.get('T', 0.0)

        observation = self._get_observation()
        info = self._get_info()
        if self.simulation_mode:
            info["simulation_elapsed_time"] = self.simulation_elapsed_time

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_observation(self):
        observation_values = [self.current_state.get(name, 0.0) for name in self.observation_variable_names]
        return np.array(observation_values, dtype=np.float32)

    def _get_info(self):
        return {
            "time": self.current_state.get('t', 0.0), # Report 't' from current_state
            "global_time_remaining": self.current_state.get('T', 0.0), # Report 'T' from current_state
            "state": dict(self.current_state)
        }

    def _render_frame(self):
        # This method is called by reset() and step() for human rendering,
        # and by render() for both human and rgb_array.
        if self.render_mode == "human":
            print(f"Step {self.current_step_count:3d} | t: {self.current_state.get('t', 0.0):.2f} | T: {self.current_state.get('T', 0.0):.2f} | "
                  f"xr: {self.current_state.get('xr', 0.0):.2f}, vr: {self.current_state.get('vr', 0.0):.2f}, ar: {self.current_state.get('ar', 0.0):.2f} | "
                  f"xg: {self.current_state.get('xg', 0.0):.2f}")
        elif self.render_mode == "rgb_array":
            # For 'rgb_array' mode, return a dummy image for now
            # Actual implementation would involve drawing the environment state
            return np.zeros((100, 100, 3), dtype=np.uint8)
        # If render_mode is None or unsupported, do nothing.

    def render(self):
        # The public render method just calls the internal _render_frame
        # which handles the different modes.
        return self._render_frame()

    def close(self):
        pass
