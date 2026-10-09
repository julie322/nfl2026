# Live First-Down Probability Model and Visualization

1. **Motivation:** Traditional stats judge a pass play only by its result (completion, incompletion, sack), so the few seconds in the pocket, where blockers, rushers and receivers decide that result, are unmeasured.
2. **Model:** Two gradient-boosted decision-tree models predict the chance that a play gains a first down: the first uses only the pre-snap situation (down, distance, field position, clock, score, personnel), and the second updates that estimate 10 times per second from player tracking (rusher distance and closing speed, pocket size, blocker retreat, receiver separation).
3. **Accuracy:** On games the model never saw in training (8,011 pass plays, 2021 Weeks 1–8), AUC rises from 0.64 before the snap to 0.71 at the throw or sack (i.e., watching the play unfold makes the model noticeably better at telling which plays will succeed).
4. **Visualization and new stat:** `play_viewer.html` animates any play with this probability curve beneath the field, and the swing from snap to throw (ΔP) becomes a new stat that credits what happens after the snap, before the ball's fate is known.
5. **Users:** Coaches can use it for film review and training, scouts to rank linemen, rushers and QBs, broadcasters as a live "first-down meter," and sportsbooks for in-play betting markets (given a live tracking feed).
