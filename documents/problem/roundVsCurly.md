The problem you're encountering with the controlFunc in "IJRR17/Theorem 14: Reach waypoint with deadline" stems from how your parser handles grouping parentheses () within the hybrid program's definition.

Let's break down the controlFunc structure in your problematic file:

keymaerax
HP controlFunc ::= {
    ( // Outer grouping parentheses
        ? (xr > waypointStartDist(xg)) ; // Test statement
        ( // Inner grouping parentheses
            ar := -b
            ++
            (?vr = 0 ; ar := 0) // Another inner grouping with sequence and test
        )
    )
    ++ // Top-level choice operator
    ( // Second outer grouping parentheses
        ? (!(xr > waypointStartDist(xg))) ;
        (
            ( // Inner grouping
                ? (xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax) ;
                ar := A
            )
            ++ // Inner choice operator
            ( // Another inner grouping
                ? (!(xr + stopDist(vr) + accComp(vr) < waypointEndDist(xg) & vr+A*ep <= Vmax)) ;
                ar := 0
            )
        )
    )
};

The key issues in your current keymaerax_parser.py that lead to the "Unhandled statement type in controlFunc parsing: (" warnings are:

split_by_operator_outside_braces does not track parentheses ():
Your split_by_operator_outside_braces function is designed to split a string by an operator only if that operator is not enclosed within curly braces {}. However, KeYmaera X uses both {} and () for grouping. In your controlFunc, the top-level ++ operator is enclosed within an outermost pair of () (which are themselves inside the HP ::= { ... } block). Because your splitter only tracks brace_level and not paren_level, it incorrectly perceives the ++ as being "inside" a grouping construct (the (...)), and thus doesn't split by it. This results in the entire ( ... ) ++ ( ... ) being treated as a single "statement" starting with (.

_parse_program_fragment does not strip outermost parentheses ():
Your _parse_program_fragment has a while loop at the beginning that strips only outermost curly braces {}. It does not strip outermost parentheses (). This means that when a fragment like ( ? (cond) ; (action) ) is passed to it, the outer parentheses remain, leading to ASTNode("Statement", value="( ? (cond) ; (action) )") or similar, which is not what you want. The parser needs to peel off any outermost matching pair of () or {}.

Test node value includes parentheses:
When _parse_program_fragment encounters a ? (test) statement, it extracts the condition as fragment[1:].strip(). If the condition itself is wrapped in parentheses (e.g., ?(xr > waypointStartDist(xg))), the Test node's value will be (xr > waypointStartDist(xg)) instead of just xr > waypointStartDist(xg).

The Theorem 1: Static safety file parses correctly because its controlFunc structure is simpler and primarily uses curly braces {} for grouping, or its parenthesized expressions are simple enough not to interfere with the ++ and ; splitting logic that only considers {}.