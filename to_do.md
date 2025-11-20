Accomplishment
- Fixed the parsing error 
- Correctly output the AST for the first file

Problem
- There is still hard code
    # Parse the loop structure: { ... } { ... } *@invariant(loopinv)
    loop_match = re.search(
        r"(\{\s*(\{\s*.*?\s*\})?\s*(\{\s*.*?\s*\})\s*\})\*@invariant\(([\s\S]*?\))\)", 
        program_raw, 
        re.DOTALL
    )
- Might fail on other kyxs file

Next step:
- Create a demo Python AST 
- Building test cases with other kyx file
- Refractor code

