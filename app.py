# filename: lottery_optimizer_app.py
import streamlit as st
import pandas as pd
import numpy as np
import itertools
import time

st.title("🎯 FAST Lottery Optimizer")

# Step 1: Upload Excel
uploaded_file = st.file_uploader("Upload your lottery Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"Loaded {len(df)} tickets")
    
    # Preprocess tickets
    df.columns = ["Ticket", "Category", "Buyer"]
    tickets = np.array([[int(x) for x in t.split(",")] for t in df["Ticket"]])
    categories = df["Category"].values
    buyers = df["Buyer"].values
    
    ALLOW_REPEAT = st.selectbox("Allow repeat digits?", ["unique", "repeat", "single"])
    
    PAYOUTS = {
        "Straight": {3: 0, 4: 18500},
        "Rumble": {3: 75, 4: 1750},
        "Chance": {1: 15, 2: 100, 3: 1100, 4: 7500}
    }
    
    # Generate draws
    digits = np.arange(10)
    draws = []
    if ALLOW_REPEAT == "unique":
        draws = list(itertools.permutations(digits, 4))
    elif ALLOW_REPEAT == "repeat":
        draws = list(itertools.product(digits, repeat=4))
    elif ALLOW_REPEAT == "single":
        for a in digits:
            for b in digits:
                for c in digits:
                    for d in digits:
                        draw = [a, b, c, d]
                        if len(set(draw)) == 3 and any(draw.count(x) == 2 for x in draw):
                            draws.append(tuple(draw))
    st.write(f"🔢 Total draws to test: {len(draws):,}")
    
    # Matching functions
    def straight_payout(draws, tickets):
        match_matrix = (draws[:, None, :] == tickets[None, :, :]).sum(axis=2)
        payout = np.zeros_like(match_matrix, dtype=int)
        payout[match_matrix == 4] = 18500
        return payout.sum(axis=1)
    
    def rumble_payout(draws, tickets):
        match_matrix = np.array([[len(set(d) & set(t)) for t in tickets] for d in draws])
        payout = np.zeros_like(match_matrix, dtype=int)
        payout[match_matrix == 3] = 75
        payout[match_matrix == 4] = 1750
        return payout.sum(axis=1)
    
    def chance_payout(draws, tickets):
        payout = np.zeros(len(draws), dtype=int)
        for i, draw in enumerate(draws):
            draw_rev = draw[::-1]
            for ticket in tickets:
                t_rev = ticket[::-1]
                match_count = 0
                for d, t in zip(draw_rev, t_rev):
                    if d == t:
                        match_count += 1
                    else:
                        break
                if match_count > 0:
                    payout[i] += PAYOUTS["Chance"].get(match_count, 0)
        return payout
    
    # Run optimization
    start = time.time()
    draws_np = np.array(draws)
    
    tickets_straight = tickets[categories == "Straight"]
    tickets_rumble = tickets[categories == "Rumble"]
    tickets_chance = tickets[categories == "Chance"]
    
    straight_payouts = straight_payout(draws_np, tickets_straight)
    rumble_payouts = rumble_payout(draws_np, tickets_rumble)
    chance_payouts = chance_payout(draws_np, tickets_chance)
    
    total_payouts = straight_payouts + rumble_payouts + chance_payouts
    
    winners = [(total_payouts[i], tuple(draws_np[i])) for i in range(len(draws_np))]
    top10 = sorted(winners, key=lambda x: x[0])[:10]
    
    st.subheader("🏆 TOP 10 LOWEST-PAYOUT DRAWS")
    for rank, (payout, draw) in enumerate(top10, 1):
        st.write(f"{rank}. Draw: {','.join(map(str, draw))} → Total payout: ₹{payout:,}")
    
    st.write(f"⏱ Completed in {time.time()-start:.2f} seconds")
