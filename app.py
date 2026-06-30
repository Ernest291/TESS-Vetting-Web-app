import streamlit as st
import lightkurve as lk
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


@st.cache_data(show_spinner="Preparing centroids...")
def get_centroid_data(tic_id, sector):
    lc = lk.search_lightcurve(f"TIC {int(tic_id)}", sector=int(sector)).download()
    if lc is None:
        return None
    lc = lc.normalize()
    time = np.asarray(lc.time.value, dtype=np.float64)
    mom_centr2 = np.asarray(lc.mom_centr2.value, dtype=np.float64)
    mom_centr1 = np.asarray(lc.mom_centr1.value, dtype=np.float64)
    pos_corr2 = np.asarray(lc.pos_corr2.value, dtype=np.float64)
    pos_corr1 = np.asarray(lc.pos_corr1.value, dtype=np.float64)

    df_centroid_x_b = pd.DataFrame(
        {"time": time, "centroid_x_b": mom_centr2}
    )  # Centroid for x/brightness
    df_centroid_x_s = pd.DataFrame(
        {"time": time, "centroid_x_s": pos_corr2}
    )  # Centroid for x/satellite motion
    df_centroid_y_b = pd.DataFrame(
        {"time": time, "centroid_y_b": mom_centr1}
    )  # Centroid for y/brightness
    df_centroid_y_s = pd.DataFrame(
        {"time": time, "centroid_y_s": pos_corr1}
    )  # Centroid for y/satellite motion
    return (
        df_centroid_x_b.dropna(),
        df_centroid_x_s.dropna(),
        df_centroid_y_b.dropna(),
        df_centroid_y_s.dropna(),
    )


@st.cache_data(show_spinner="Preparing TPF Tests...")
def get_tpf_data(tic_id, sector, t0):
    tpf = lk.search_targetpixelfile(f"TIC {int(tic_id)}", sector=int(sector)).download()
    if tpf is None:
        return None
    tpf_list = [tpf.flux.value]
    t_list = [tpf.time.value]
    t0_list = [t0]
    return (tpf_list, t_list, t0_list)


# 2. streamlit layout

with st.sidebar:
    with st.form(key="sidebar_input"):
        tic_id = st.number_input(label="TIC ID: ", format="%.0f")
        sector = st.number_input(label="Sector: ", format="%.0f")
        transit_time = st.number_input(label="Transit Time: ")
        bg_flux = st.checkbox(label="check bg flux?")
        centroid_plot = st.checkbox(label="check Centroid Plot?")
        in_out_transit_diff = st.checkbox(
            label="check In-Out Transit difference image?"
        )
        pixel_level = st.checkbox(label="check Pixel-Level Plot?")

        submitted = st.form_submit_button()

if submitted:
    if not tic_id or not sector:  # if incomplete input then raise error
        st.error("Invalid Input. Make sure to fill in TIC ID and/or Sector.")
    else:  # render plots and output
        st.balloons()
        # st.write("### Output")
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
        # -------------------------------------------------------------------------------------------------
        #                                       VETTING TESTS
        # -------------------------------------------------------------------------------------------------

        # Background Flux section--------------------------------------------------------------------------
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

            # Centroid Positions ------------------------------------------------------------------------------

            if centroid_plot:
                st.divider()
                cs = get_centroid_data(tic_id, sector)
                if cs is not None:
                    cxb = cs[0]
                    cxs = cs[1]
                    cyb = cs[2]
                    cys = cs[3]

                    # I just found out that you need to normalize the data lol :0
                    def normalize(data):
                        return (data - np.min(data)) / (np.max(data) - np.min(data))

                    fig3 = go.Figure()
                    fig3.add_trace(
                        go.Scattergl(
                            x=cs[0]["time"],
                            y=normalize(cs[0]["centroid_x_b"]),
                            mode="lines",
                            line=dict(color="orange", width=2),
                            name="brightness motion",
                        )
                    )

                    fig3.add_trace(
                        go.Scattergl(
                            x=cs[1]["time"],
                            y=normalize(cs[1]["centroid_x_s"]),
                            mode="lines",
                            line=dict(color="white", width=2),
                            name="satellite motion",
                        )
                    )
                    if transit_time != 0:
                        fig3.add_vline(
                            x=transit_time,
                            line_dash="dash",
                            line_color="darkorange",
                            line_width=3,
                            annotation_text="t0",
                            annotation_position="top left",
                        )

                    fig3.update_layout(
                        title=f"TIC {int(tic_id)} Centroid Plot (x-axis)",
                        yaxis_title="Centroid Positions (x-axis)",
                        xaxis_title="Time (BTJD)",
                        template="plotly_dark",
                        font=dict(size=18),
                    )

                    st.plotly_chart(fig3, use_container_width=True)

                    fig4 = go.Figure()
                    fig4.add_trace(
                        go.Scattergl(
                            x=cs[2]["time"],
                            y=normalize(cs[2]["centroid_y_b"]),
                            mode="lines",
                            line=dict(color="orange", width=2),
                            name="brightness motion",
                        )
                    )

                    fig4.add_trace(
                        go.Scattergl(
                            x=cs[3]["time"],
                            y=normalize(cs[3]["centroid_y_s"]),
                            mode="lines",
                            line=dict(color="white", width=2),
                            name="satellite motion",
                        )
                    )
                    if transit_time != 0:
                        fig4.add_vline(
                            x=transit_time,
                            line_dash="dash",
                            line_color="darkorange",
                            line_width=3,
                            annotation_text="t0",
                            annotation_position="top left",
                        )

                    fig4.update_layout(
                        title=f"TIC {int(tic_id)} Centroid Plot (y-axis)",
                        yaxis_title="Centroid Positions (y-axis)",
                        xaxis_title="Time (BTJD)",
                        template="plotly_dark",
                        font=dict(size=18),
                    )
                    st.plotly_chart(fig4, use_container_width=True)

                else:
                    st.write("debug it!")

            # In-Out Transit difference -----------------------------------------------------------------------

            if in_out_transit_diff:
                st.divider()
                st.write("Work in Progress")
                if transit_time != 0:
                    tpfs = get_tpf_data(tic_id, sector, transit_time)

                    # This particular snippet is made by Nora Eisner: https://github.com/noraeisner/PH_Coffee_Chat
                    if tpfs is not None:
                        tpflist = tpfs[0]  # tpf_list
                        tlist = tpfs[1]  # t_list
                        t0list = tpfs[2]  # t0_list

                        fig = plt.figure(figsize=(9, 2.5 * len(t0list)))
                        plt.tight_layout()

                        count = 0

                        for idx, tpf_filt in enumerate(tpflist):
                            T0 = t0list[idx]
                            t = tlist[idx]

                            intr = (
                                abs(T0 - t) < 0.25
                            )  # Create a mask of the in-transit times
                            ootr = (abs(T0 - t) < 0.5) * (
                                abs(T0 - t) < 0.3
                            )  # Create a mask of the out-of-transit times
                            img_intr = tpf_filt[intr, :, :].sum(axis=0) / float(
                                intr.sum()
                            )  # Apply the masks and normalize the flux
                            img_ootr = tpf_filt[ootr, :, :].sum(axis=0) / float(
                                ootr.sum()
                            )
                            img_diff = (
                                img_ootr - img_intr
                            )  # Calculate the difference image

                            count += 1
                            plt.subplot(len(t0list), 3, count)
                            plt.axis("off")
                            plt.imshow(img_intr, cmap=plt.cm.viridis, origin="lower")
                            plt.colorbar()
                            plt.title(
                                "t = {} days \n In Transit Flux (e-/cadence)".format(
                                    T0
                                ),
                                fontsize=9,
                            )

                            count += 1
                            plt.subplot(len(t0list), 3, count)
                            plt.axis("off")
                            plt.imshow(img_ootr, cmap=plt.cm.viridis, origin="lower")
                            plt.colorbar()
                            plt.title("Out of Transit Flux (e-/cadence)")

                            count += 1
                            plt.subplot(len(t0list), 3, count)
                            plt.axis("off")
                            plt.imshow(img_diff, cmap=plt.cm.viridis, origin="lower")
                            plt.colorbar()
                            plt.title("Difference Image Flux (e-/cadence)", fontsize=9)

                        plt.subplots_adjust(wspace=0)
                        plt.tight_layout()
                        st.pyplot(fig)

                        # ----------------------------End of Snippet------------------------------------------
                else:
                    st.write("Transit Time is needed to plot this!")
