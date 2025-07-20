
def get_prompt_family():
    prompt = """ You are an expert logician specializing in solving complex family tree puzzles. Your task is to analyze familial relationships, deduce unknown connections, and identify contradictions to determine the correct answer. You will use a Tree of Thoughts (ToT) approach, meticulously building the family tree and then evaluating each given option against your constructed tree and all stated clues.
    
    **Your Process (Tree of Thoughts for Family Trees):**
    
    1.  **Consolidated Clue Analysis:** List all provided clues. For each clue, infer immediate, direct relationships (e.g., Parent-Child, Spousal, Sibling). Note any gender implications.
    2.  **Initial Tree Construction:** Begin to build a skeleton family tree based on the most foundational or interlinked clues. Use symbols for relationships (e.g., `=` for spouse, `-` for sibling, `|` for parent-child) and denote genders (M/F) or unknown (`?`). Identify the total number of members as a crucial constraint.
    3.  **Iterative Deduction & Integration:** Systematically integrate remaining clues into the evolving tree. For each integration:
        * **Propose a connection:** How does this clue fit into the current tree?
        * **Test consistency:** Does this connection contradict any *previously established* part of the tree or *any other clue*?
        * **Refine tree:** Update the tree with new members, relationships, or resolved genders. If a contradiction arises, explicitly state it and backtrack if necessary, exploring alternative interpretations of how clues integrate.
    4.  **Final Tree Confirmation:** Once all clues are integrated, review the complete family tree. Ensure it meets all conditions, especially the total member count.
    5.  **Option Evaluation (ToT Branches):** For each given option, assume it is true. Then, verify if this assumed relationship is consistent with your **final confirmed family tree** and all original clues.
        * **For Consistent Options:** State why it fits all criteria.
        * **For Inconsistent Options:** Clearly state the specific contradiction found with the family tree or the original clues.
    6.  **Final Answer:** State the definitively correct option.
    
    **Constraints:**
    * Be rigorous and systematic.
    * Clearly label each step of your process.
    * Explicitly state contradictions for inconsistent paths/options.
    * Ensure the final family tree adheres to the exact member count.
    
    ---
    **Problem:**
    
    In an eight member family, L is the daughter – in – law of P. X is the uncle of F who is the grandchild of J. V is not the spouse of P. J is the father – in – law of R. P is the mother of H. R and X are not siblings. F is the niece of H. X is the only son of V. Both H and R have same gender. How is H related to R ?
    
    **Options:**
    A) Sister
    B) Brother in Law
    C) Father
    D) Brother
    
    **ToT Analysis for Family Tree Problem:**
    
    **Step 1: Consolidated Clue Analysis**
    
    * **Clue 1:** L is the daughter-in-law of P. (L is married to P's son). P(F) -> Son (?) = L(F).
    * **Clue 2:** X is the uncle of F.
    * **Clue 3:** F is the grandchild of J.
    * **Clue 4:** V is not the spouse of P.
    * **Clue 5:** J is the father-in-law of R. (J(M) -> Child (?) = R(?)).
    * **Clue 6:** P is the mother of H. P(F) -> H(?).
    * **Clue 7:** R and X are not siblings.
    * **Clue 8:** F is the niece of H. (F is child of H's sibling).
    * **Clue 9:** X is the only son of V. V(?) -> X(M, only son).
    * **Clue 10:** Both H and R have same gender.
    * **Clue 11:** Eight member family (total count = 8).
    
    **Step 2: Initial Tree Construction & Step 3: Iterative Deduction & Integration**
    
    Let's integrate clues to build the tree.
    
    * From (6) & (8): P(F) -> H(?), Sibling (let's call 'S') (?). Sibling (S) -> F(?).
    * From (1) & (8) & (10): L is daughter-in-law of P. F is niece of H. This means F is child of H's sibling (S). L must be married to S if S is male.
        * Let's assume S is the person 'R' from clue 5. So R(M) = L(F). P(F) -> H(?), R(M).
        * Clue 10: H and R have same gender. Since R is Male (married to L), **H must be Male**.
        * **Current Tree Segment 1:**
            ```
            P (F)
             |
            H (M) - R (M) = L (F)
                      |
                      F (?)
            ```
    * From (3) & (5) & (10): F is grandchild of J. J is father-in-law of R.
        * R is married to L. So J is father of L. J(M) -> L(F).
        * This means F is child of R & L. J is L's father. So F is J's grandchild. Consistent.
        * **Current Tree Segment 2 (Integrated):**
            ```
                      J (M)
                       |
            P (F) --- L (F) = R (M)
             |              |
            H (M)           F (?)
            ```
            (Note: P and J are connected through their children R and L marrying)
    
    * From (2) & (9): X is uncle of F. X is only son of V.
        * F is child of R & L. X is uncle of F, so X is sibling of R or L.
        * If X is sibling of R: X is P's child. X is only son of V. This implies V is spouse of P.
            * **Check Clue 4:** V is not spouse of P. **Contradiction.** So X is not R's sibling.
        * Therefore, X is **L's sibling**.
        * Since X is L's sibling, and L is J's child, X is also J's child.
        * X is the only son of V. Since J is M and father of L & X, **V must be J's spouse and X's mother**.
        * **Current Tree Segment 3 (Integrated):**
            ```
               J (M) = V (F)
                 |
               L (F) - X (M, only son)
                 |
            P (F) = R (M)
             |
            H (M) - (implied sibling)
                      |
                      F (?)
            ```
            (Correcting the horizontal line: R is P's son, L is J's daughter, they are married. H is P's son. X is J's son.)
    
    * **Refined Combined Family Tree:**
        ```
               J (M) = V (F)
                 |
               L (F) - X (M)
                 |
                 = R (M)
                 |
               P (F)
                 |
               H (M)
                 |
               F (?)
        ```
        (This is clearer representation)
        Let's check the lines again:
        P is mother of H and R. R is married to L. L is daughter of J and V. X is son of J and V. F is child of R and L.
    
        **Final Confirmed Family Tree Structure:**
        ```
               (Father of P - not in family)
                 |
               P (F) -------- (Father of H & R - not in family/deceased)
                 |
               H (M) --- R (M)  <-- Siblings
                           |
                           L (F) = R (M)
                           |
                           F (?)
                           |
                   J (M) = V (F)   <-- Married Couple
                     |
                     L (F) --- X (M) <-- Siblings (L is F, X is M)
        ```
        This means the relationships are:
        * P is mother of H and R.
        * J and V are parents of L and X.
        * R (P's son) is married to L (J & V's daughter).
        * F is the child of R and L.
    
    **Step 4: Final Tree Confirmation & Member Count**
    
    * **P (F):** 1
    * **H (M):** 2
    * **R (M):** 3
    * **L (F):** 4
    * **J (M):** 5
    * **V (F):** 6
    * **F (?):** 7
    * **X (M):** 8
    
    Total members = 8. All members are accounted for.
    
    **Verify All Clues with Final Tree:**
    
    1.  L is the daughter-in-law of P. (Yes, L married R, P's son).
    2.  X is the uncle of F. (Yes, X is L's brother, so F's maternal uncle).
    3.  F is the grandchild of J. (Yes, F is L's child, J is L's father).
    4.  V is not the spouse of P. (Yes, V is spouse of J).
    5.  J is the father-in-law of R. (Yes, R married L, J's daughter).
    6.  P is the mother of H. (Yes).
    7.  R and X are not siblings. (Yes, R is P's son, X is J&V's son. They are brothers-in-law).
    8.  F is the niece of H. (Yes, F is R's child, H is R's brother).
    9.  X is the only son of V. (Yes, consistent within the J-V branch).
    10. Both H and R have same gender. (Yes, both Male).
    11. Eight members. (Yes, 8 members identified).
    
    The family tree is fully consistent.
    
    **Step 5: Option Evaluation (Against Final Tree)**
    
     **A) Sister**
        * **Assumption:** H is Female and R is Male (or Female).
        * **Check with Tree:** Our final tree shows H as Male.
        * **Contradiction:** Gender of H does not match.
        * **Result:** **INCONSISTENT.**
    
     **B) Brother in Law**
        * **Assumption:** H is R's Brother-in-Law.
        * **Check with Tree:** Our final tree shows H is R's direct brother (sharing parent P).
        * **Contradiction:** H is a direct brother, not brother-in-law.
        * **Result:** **INCONSISTENT.**
    
     **C) Father**
        * **Assumption:** H is Father of R.
        * **Check with Tree:** Our final tree shows H is R's brother (sharing parent P).
        * **Contradiction:** H is R's brother, not father.
        * **Result:** **INCONSISTENT.**
    
     **D) Brother**
        * **Assumption:** H is Brother of R.
        * **Check with Tree:** Our final tree shows H is Male, R is Male, and they share parent P. This confirms they are brothers.
        * **Result:** **CONSISTENT.**
    
    **Step 6: Final Answer**
    
    The only option consistent with the fully constructed and verified family tree is D.
    
    **The relation is D) Brother.**
    Returns "B"
    """
    return prompt

    
def get_prompt_arrange():
    prompt = """
    You are an expert in logical reasoning and arrangement puzzles. Your task is to solve Circular/Linear Arrangement problems by systematically building the arrangement based on given clues and then validating statements or options against your derived arrangement. You will employ a Tree of Thoughts (ToT) approach, where you explore possible arrangements, identify contradictions, and converge on the correct one.

**Your Process (Tree of Thoughts for Arrangements):**

1.  **Consolidated Clue Analysis:**
    * Identify direct relationships from all the given clues (e.g., "immediate left," "opposite").
    * Identify numerical spacing (e.g., "two persons between").
    * Identify negative constraints (e.g., "not a neighbor").
    * Note the total number of persons and the type of arrangement (circular/linear, facing direction).

2.  **Anchor & Initial Placement (Branching - if ambiguous):**
    * Start with the most definitive clue or a clue that links multiple persons.
    * For circular arrangements, pick an arbitrary starting position for one person if there's no absolute reference.
    * If a clue allows for multiple initial arrangements (e.g., "A is 3rd to the right of B" can have B in different spots), create separate branches for each plausible starting setup.
    Systematically place other persons based on remaining clues, prioritizing those that connect to already placed persons.
    * For each placement:
        * **Propose a position:** Where can the next person sit?
        * **Test consistency:** Does this placement contradict any *already placed persons* or *any original clue* (even those not yet used for placement)?
    * Keep track of filled and empty seats.

    3.  **Option Evaluation (ToT Branches for Solutions):**
    * For each given option/statement (A, B, C, D...):
        * **Assume the option is true.**
        * **Check with Final Arrangement:** Does this statement align with your derived final arrangement?
        * **For Consistent Options:** State why it matches the final arrangement.
        * **For Inconsistent Options:** Clearly state the specific contradiction with your final arrangement.
    * Identify the option that is definitively correct based on your consistent arrangement.


4.  **Final Arrangement Confirmation:**
    * Once all persons are placed, review the complete arrangement.
    * Ensure every single original clue is satisfied by the final arrangement. If not, backtrack and re-evaluate earlier decisions.



**Constraints:**
* Be rigorous and systematic in placing persons.
* Clearly label each step and assumption.
* Explicitly state all contradictions and pruned paths.
* Provide the final, unique consistent arrangement.
* MAke sure to return the correct OPTION NOT the correct answer eg if B is the correct answer and that is opton D) return D

**Question:**

Eight persons P, Q, R, S, T, U, V and W are sitting around a circular table facing the center (not necessarily in the same order).

* Two persons sit between Q and W.
* V sits immediate left of W.
* Three persons sit between S and T.
* R sits immediate left of T.
* U is not the neighbor of P.
* One person sits between P and Q.

Which of the following statements is correct?
A) Q sits to the immediate right of U.
B) One person sits between P and R.
C) Three persons sit between Q and R.
D) One person sits between S and U.

**Answer:** D) One person sits between S and U.

**Tree of Thoughts to reach solution**

**ToT Analysis for Circular Sitting Arrangement Problem:**

**1. Clue Summary:**
* **Persons:** P, Q, R, S, T, U, V, W (8 total)
* **Arrangement:** Circular table.
* **Clues:**
    * (1) 2 persons between Q & W.
    * (2) V immediate left of W.
    * (3) 3 persons between S & T. (S & T opposite)
    * (4) R immediate left of T.
    * (5) U not neighbor of P.
    * (6) 1 person between P & Q.

**2. Arrangement Construction (Step-by-step Deduction):**

* **Anchor:** Let's place W.
    * `W` (position 1)

* **Sure (from Clue 2):** V sits immediate left of W.
    * `V` (position 2, clockwise from W)
    * Current: `(W, V, _, _, _, _, _, _)`

* **Maybe (from Clue 1):** Two persons between Q and W.
    * **Path A: Q is 3rd clockwise from W.**
        * Q at position 4. `(W, V, _, Q, _, _, _, _)`
        * **Sure (from Clue 6):** One person sits between P and Q. (Q at 4). Seat 2 (V) is taken. P must be at position 6.
        * Current: `(W, V, _, Q, _, P, _, _)` (Remaining: 3, 5, 7, 8)
        * **Maybe (from Clues 3 & 4):** S and T are opposite (3 persons between them), R is immediate left of T.
            * Possible T/R placements in remaining seats:
                * If T at 3, R at 4. **IMPOSSIBLE:** Q is at 4.
                * If T at 5, R at 6. **IMPOSSIBLE:** P is at 6.
                * If T at 7, R at 8. This is **SURE** for T/R.
                    * Place T at 7, R at 8.
                    * S (opposite T) must be at 3.
                    * Remaining seat 5 is U.
                    * Arrangement: `(W, V, S, Q, U, P, T, R)`
                * **Check (from Clue 5):** U is not the neighbor of P. In `(W, V, S, Q, U, P, T, R)`, U (pos 5) and P (pos 6) are neighbors. **IMPOSSIBLE:** Contradicts Clue 5.
        * **Prune Path A.**

    * **Path B: Q is 3rd counter-clockwise from W.**
        * Q at position 6. `(W, V, _, _, _, Q, _, _)`
        * **Sure (from Clue 6):** One person sits between P and Q. (Q at 6). P can be at 4 or 8.
            * **Sub-Path B1: P at position 4.**
                * Current: `(W, V, _, P, _, Q, _, _)` (Remaining: 3, 5, 7, 8)
                * **Maybe (from Clues 3 & 4):** S and T opposite, R immediate left of T.
                    * Possible T/R placements in remaining seats:
                        * If T at 3, R at 4. **IMPOSSIBLE:** P is at 4.
                        * If T at 7, R at 8. This is **SURE** for T/R.
                            * Place T at 7, R at 8.
                            * S (opposite T) must be at 3.
                            * Remaining seat 5 is U.
                            * Arrangement: `(W, V, S, P, U, Q, T, R)`
                        * **Check (from Clue 5):** U is not the neighbor of P. In `(W, V, S, P, U, Q, T, R)`, U (pos 5) and P (pos 4) are neighbors. **IMPOSSIBLE:** Contradicts Clue 5.
                * **Prune Sub-Path B1.**

            * **Sub-Path B2: P at position 8.**
                * Current: `(W, V, _, _, _, Q, _, P)` (Remaining: 3, 4, 5, 7)
                * **Maybe (from Clues 3 & 4):** S and T opposite, R immediate left of T.
                    * Possible T/R placements in remaining seats:
                        * If T at 3, R at 4. This is **SURE** for T/R.
                            * Place T at 3, R at 4.
                            * S (opposite T) must be at 7.
                            * Remaining seat 5 is U.
                            * Arrangement: `(W, V, T, R, U, Q, S, P)`
                        * **Check (from Clue 5):** U is not the neighbor of P. In `(W, V, T, R, U, Q, S, P)`, U (pos 5) and P (pos 8) are NOT neighbors. **SURE:** Consistent with Clue 5.
                * **This path is consistent!**

**Final Confirmed Arrangement:** (Clockwise from W)
`(W, V, T, R, U, Q, S, P)`

**3. Option Evaluation:**

* **A) Q sits to the immediate right of U.**
    * In `(..., R, U, Q, ...)`, Q is immediate left of U.
    * **Result:** **IMPOSSIBLE.** (Contradicts arrangement).

* **B) One person sits between P and R.**
    * P is at 8, R is at 4. Clockwise from P: W, V, T (3 persons). Counter-clockwise from P: S, Q, U (3 persons).
    * **Result:** **IMPOSSIBLE.** (Three persons sit between P and R).

* **C) Three persons sit between Q and R.**
    * Q is at 6, R is at 4. Clockwise from Q: S, P, W, V, T (5 persons). Counter-clockwise from Q: U (1 person).
    * **Result:** **IMPOSSIBLE.** (One person (U) sits between Q and R on shortest path).

* **D) One person sits between S and U.**
    * S is at 7, U is at 5. Clockwise from S: P, W, V, T, R (5 persons). Counter-clockwise from S: Q (1 person).
    * **Result:** **SURE.** (One person (Q) sits between S and U).
    Returns "D"
    """
    return prompt
def get_prompt_truth():
    prompt = """
        You are an expert logician specializing in solving truth and lie puzzles using a Tree of Thoughts (ToT) approach. Your task is to analyze complex statements, generate hypotheses, trace their logical implications, identify contradictions, and ultimately deduce the truth.
    
    **Your Process (ToT Steps):**
    1. If the number of possibilities involved in this question are not very like ( say 2^8) you muse use exhaustion of possibilities to solve it. List all possibilities and for the given information , check each option and which ones contradict the given information
    2. If the number of possibilities is huge , you must generate a Tree of thoughts where you create a map of the information you have and check with each option if it aligns with your information. If it does, explore that branch, if contraction is emcountered ,prune thar branch and do not explore further
    
    **Constraints:**
    * Be rigorous in your logical deduction.
    * Clearly label each hypothesis and its associated deductions/contradictions.
    * Focus on efficient pruning of impossible paths.
    *  MAKE sure to return the correct OPTION NOT the correct answer eg if B is the correct answer and that is opton D) return D

    Here is an example :
    **Question:**: Ramesh, Suresh and Mahesh are three people who belong to three different tribes of people. The three tribes are known as knights (those who always speak the truth), Knaves(who always lie) and Alters(those who alternatively speak the truth and lie).

Ramesh said that Suresh is not an alter. Mahesh said that Ramesh is an alter.
Who among the following is Knave?



**Options:**
A) Ramesh
B) Suresh
C) Mahesh
D) cannot be determined

**Answer:** B) Suresh

    ToT Analysis Solving by Exhaustion:



**1. Problem Setup & Clue Analysis:**
* **Individuals:** Ramesh , Suresh , Mahesh
* **Tribes:** Knight (Truth teller), Knave (Liar), Alters (Alternates)
* **Statements:**
Ramesh claims : Suresh is not an alter 
Mahesh claims : Ramesh is an alter 
* **Global Constraints:** They all belong to different tribes so no two people can be Knights , Knaves or Alters.

**2. Hypothesis Generation & Validation (ToT Branches):**

Since we only have 3 people, we can list down the possible cases.
Let T denote the knights, L denote the knaves, and A denotes the alters.

Then possible arrangements are
TLA, TAL, ATL, ALT, LAT, LTA
Ramesh said that Suresh is not an alter, so we can remove the cases TAL and LTA. This is because if Ramesh is a knight then Suresh can not be an alter and if Ramesh is a knave than Suresh is an alter. So we have 4 cases left which are

Mahesh says that Ramesh is an alter, so using this statement we can rule out cases LAT and ATL.
So we have two cases which are left these are TLA and ALT
From these cases, Ramesh can either be a knight or an alter, Suresh is a knave and Mahesh can also be either knight or an alter.
TLA, ATL, ALT,LAT
Mahesh says that Ramesh is an alter, so using this statement we can rule out cases LAT and ATL.
So we have two cases which are left these are TLA and ALT
In both the cases, Suresh is the Knave. So option A is correct
**5. Final Answer:**
The correct answer is **B) Suresh**.

    Returns "B"

    
    """
    return prompt
