from __future__ import annotations

import re

Number = r"(\d+)"
Fraction = rf"{Number}\s*/\s*{Number}"
Constant = r"([a-z_][a-zA-Z0-9_-]*)"
Variable = r"([A-Z][a-zA-Z0-9_-]*)"
Literal = rf"({Constant}|{Variable})"
Triple = rf"{Constant}\s*\(\s*{Literal}\s*,\s*{Literal}\s*\)"
Body = rf"{Triple}\s*(,\s*{Triple})*"
Rule = rf"{Triple}\s*:\s*{Fraction}\s*:-\s*{Body}\s*\.\s*"



if __name__ == '__main__':
    # ptrn = re.compile(Rule)
    # match = ptrn.findall("treats(c3,X) : 2 / 4 :- associates(A2,X), participates(A2,A3), participates(A4,A3).")
    # print(match)

    print(re.match(Number, "2").groups())
    print(re.match(Number, "4").groups())
    print(re.match(Fraction, "2 / 4").groups())
    print(re.match(Constant, "treats").groups())
    print(re.match(Variable, "A2").groups())
    print(re.match(Literal, "A2").groups())
    print(re.match(Literal, "treats").groups())
    print(re.match(Triple, "treats(A2,c1)").groups())
    print(re.match(Body, "associates(A2,X), participates(A2,A3), participates(A4,A3)").groups())


    print('Done.')
