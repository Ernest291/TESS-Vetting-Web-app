import streamlit as st
import lightkurve as lk
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt 

st.title("TESTS GO HERE :)")

tpf = lk.search_targetpixelfile(
    "TIC 448567661",
    sector=102,
    author="SPOC"
).download()

image = tpf.flux[0]

nx, ny = image.shape 
xpix, ypix = np.meshgrid(np.arange(nx), np.arange(ny))
sky = tpf.wcs.pixel_to_world(xpix, ypix)
ra, dec = sky.ra.deg, sky.dec.deg 

fig = go.Figure()

fig.add_trace(go.Heatmap(
    z = image,
    x = ra[0, :],
    y = dec[:, 0],
    colorscale="Viridis"
    )
)

fig.add_trace(
        go.Scatter(
            x=[tpf.ra],
            y=[tpf.dec],
            mode="markers",
            marker=dict(size=18, symbol="star", color="red"),
            name="Target"
            )
        )

fig.update_layout(
        xaxis_title="RA (deg)",
        yaxis_title="DEC (deg)",
        height=600,
        width=300
        )
st.plotly_chart(fig, use_container_width=True)
