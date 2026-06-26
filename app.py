import streamlit as st
import lightkurve as lk
import plotly.graph_objects as go
import pandas as pd

# 1. Cache data


@st.cache_data(show_spinner="Downloading lightcurve data...")
def get_lightcurve_data(tic_id, sector):
    star = lk.search_lightcurve(f"TIC {int(tic_id)}", sector=int(sector)).download()
    if star is None:
        return None
    star = star.normalize()
    df = pd.DataFrame({"time": star.time.value, "flux": star.flux.value})
    return df.dropna()


@st.cache_data(show_spinner="Downloading targetpixelfile data...")
def get_background_data(tic_id, sector):
    tpf = lk.search_targetpixelfile(f"TIC {int(tic_id)}", sector=int(sector)).download()
    if tpf is None:
        return None
    bkg = tpf.get_bkg_lightcurve()
    df = pd.DataFrame({"time": bkg.time.value, "flux": bkg.flux.value})
    return df.dropna()


# 2. streamlit layout

with st.sidebar:
    with st.form(key="input"):
        tic_id = st.number_input(label="TIC ID: ", format="%.0f")
        sector = st.number_input(label="Sector: ", format="%.0f")
        transit_time = st.number_input(label="Transit Time: ")
        bg_flux = st.checkbox(label="check bg flux?") 
        centroid_plot = st.checkbox(label="check Centroid Plot?")
        in_out_transit_diff = st.checkbox(label="check In-Out Transit difference image?")
        pixel_level = st.checkbox(label="check Pixel-Level Plot?")

        submitted = st.form_submit_button()

if submitted:
    if not tic_id or not sector:  # if incomplete input then raise error
        st.error("Invalid Input. Make sure to fill in TIC ID and/or Sector.")
    else:  # render plots and output
        st.balloons()
        st.write("### Output")
        st.write(f"**TIC:** {int(tic_id)} | **Sector:** {int(sector)}")

        # Call the functions to fetch the data
        lc_df = get_lightcurve_data(tic_id, sector)
        if lc_df is not None:
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scattergl(
                    x=lc_df["time"],
                    y=lc_df["flux"],
                    mode="markers",
                    marker=dict(color="deepskyblue", symbol="star", size=6),
                    name="Flux",
                )
            )
            lc_df["bin"] = pd.cut(lc_df["time"], bins=300)
            binned_lc = lc_df.groupby("bin", observed=False).agg(
                {"time": "mean", "flux": "mean"}
            )

            fig1.add_trace(
                go.Scattergl(
                    x=binned_lc["time"],
                    y=binned_lc["flux"],
                    mode="lines",
                    line=dict(color="white", width=3),
                    name="Binned Flux",
                )
            )
            if transit_time != 0:
                fig1.add_vline(
                    x=transit_time,
                    line_dash="dash",
                    line_color="darkorange",
                    line_width=3,
                    annotation_text="t0",
                    annotation_position="top left",
                )

            fig1.update_layout(
                title=f"TIC {int(tic_id)} Sector {int(sector)}",
                xaxis_title="Time(BTJD)",
                yaxis_title="Normalized Flux",
                template="plotly_dark",
                font=dict(size=18),
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.error("No Lightcurve data found.")
#-------------------------------------------------------------------------------------------------
#                                       VETTING TESTS
#-------------------------------------------------------------------------------------------------

# Background Flux section
        if bg_flux:
            st.divider()
            st.write("### Vetting Tests")
            # Fetch BKG_LC data
            bg_df = get_background_data(tic_id, sector)
            if bg_df is not None:
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scattergl(
                        x=bg_df["time"],
                        y=bg_df["flux"],
                        mode="markers",
                        marker=dict(color="blue", symbol="star", size=6),
                        name="Background Flux",
                    )
                )

                bg_df["bin"] = pd.cut(bg_df["time"], bins=300)
                binned_bg = bg_df.groupby("bin", observed=False).agg(
                    {"time": "mean", "flux": "mean"}
                )

                fig2.add_trace(
                    go.Scattergl(
                        x=binned_bg["time"],
                        y=binned_bg["flux"],
                        mode="lines",
                        line=dict(color="white", width=3),
                        name="Binned BG FLux",
                    )
                )
                if transit_time != 0:
                    fig2.add_vline(
                        x=transit_time,
                        line_dash="dash",
                        line_color="darkorange",
                        line_width=3,
                        annotation_text="t0",
                        annotation_position="top left",
                    )

                fig2.update_layout(
                    title=f"TIC {int(tic_id)} Background Flux",
                    xaxis_title="Time(BTJD)",
                    yaxis_title="Unormalized Flux",
                    template="plotly_dark",
                    font=dict(size=18),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("No available bg flux data to show.")
            
#
            if in_out_transit_diff:
                st.divider()
                st.write("Work in Progress")



