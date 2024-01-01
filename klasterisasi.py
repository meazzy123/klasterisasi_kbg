import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.animation import FuncAnimation


def intro():
    st.markdown(
        """
        <div style="text-align: center; background-color: #3498db; padding: 20px; border-radius: 10px;">
            <h2 style="color: #fff;">Klasterisasi Kejahatan Berbasis Gender Terhadap Perempuan di Indonesia</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tambahkan efek animasi atau transisi CSS
    st.markdown(
        """
        <style>
            .animated-text {
                font-size: 24px;
                opacity: 0;
                animation: fadeIn 2s ease-in-out forwards;
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                }
                to {
                    opacity: 1;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Tambahkan kelas CSS ke teks untuk menerapkan efek animasi
    st.markdown(
        """
        <div class="animated-text">
            Silahkan memilih tahun yang tersedia untuk melihat hasil klasterisasinya
        </div>
        """,
        unsafe_allow_html=True,
    )


def tahun2022():
    data22 = pd.read_excel("dataset_kbg_perempuan.xlsx", sheet_name="2022")
    data22.head()

    # st.header("Isi Dataset")
    # st.write(data21)

    st.header("Dataset")
    data22["Total KBG Perempuan"] = (
        data22["Komnas Perempuan"] + data22["Data Lembaga Mitra"] + data22["Badilag"]
    )
    st.write(data22)

    # st.header("Data Training")
    x_train22 = data22[["Total KBG Perempuan"]].values
    # st.write(x_train21)

    scaler = MinMaxScaler()
    x_train22 = scaler.fit_transform(x_train22)
    # st.header("Hasil Scaling")
    # st.write(x_train21)

    inertias = []
    k_range = range(1, 10)
    for k in k_range:
        km = KMeans(n_clusters=k).fit(x_train22)
        inertias.append(km.inertia_)

    # Membuat dataframe untuk Plotly
    df_elbow = pd.DataFrame({"Number of Clusters": k_range, "Inertia": inertias})

    # Membuat plot interaktif dengan Plotly
    fig = px.line(
        df_elbow,
        x="Number of Clusters",
        y="Inertia",
        markers=True,
        title="Elbow Method For Optimal k",
        labels={"Inertia": "Sum of squared distances (Inertia)"},
    )

    # Menampilkan plot
    st.plotly_chart(fig)
    st.sidebar.subheader("Nilai Jumlah K")
    clust = st.sidebar.slider("Pilih jumlah kluster:", 2, 10, 3, 1)

    def k_means(n_clust):
        kmeans = KMeans(n_clusters=n_clust)

        y_cluster22 = kmeans.fit_predict(x_train22)
        data22["Cluster"] = y_cluster22
        colors = [plt.cm.nipy_spectral(float(i) / n_clust) for i in range(n_clust)]

        # Membuat dataframe baru dengan informasi yang diperlukan
        df_plotly = pd.DataFrame(
            {
                "Total KBG Perempuan": data22["Total KBG Perempuan"],
                "Provinsi": data22["Provinsi"],
                "Cluster": data22["Cluster"],
                "Komnas Perempuan": data22["Komnas Perempuan"],
                "Data Lembaga Mitra": data22["Data Lembaga Mitra"],
                "Badilag": data22["Badilag"],
            }
        )

        # Membuat scatter plot interaktif dengan Plotly
        fig = px.scatter(
            df_plotly,
            x="Total KBG Perempuan",
            y="Provinsi",
            color="Cluster",
            color_discrete_sequence=colors,
            labels={"Cluster": "Cluster"},
            title="Persebaran Kejahatan Berbasis Gender Terhadap Perempuan",
            hover_name="Provinsi",
            hover_data={
                "Komnas Perempuan": True,
                "Data Lembaga Mitra": True,
                "Badilag": True,
                "Total KBG Perempuan": True,
                "Provinsi": False,
                "Cluster": True,
            },
        )

        # Menampilkan plot
        st.plotly_chart(fig)
        st.write(data22)

    k_means(clust)


def tahun2021():
    data21 = pd.read_excel("dataset_kbg_perempuan.xlsx", sheet_name="2021")
    data21.head()

    # st.header("Isi Dataset")
    # st.write(data21)

    st.header("Dataset")
    data21["Total KBG Perempuan"] = (
        data21["Komnas Perempuan"] + data21["Data Lembaga Mitra"] + data21["Badilag"]
    )
    st.write(data21)

    # st.header("Data Training")
    x_train21 = data21[["Total KBG Perempuan"]].values
    # st.write(x_train21)

    scaler = MinMaxScaler()
    x_train21 = scaler.fit_transform(x_train21)
    # st.header("Hasil Scaling")
    # st.write(x_train21)

    inertias = []
    k_range = range(1, 10)
    for k in k_range:
        km = KMeans(n_clusters=k).fit(x_train21)
        inertias.append(km.inertia_)

    # Membuat dataframe untuk Plotly
    df_elbow = pd.DataFrame({"Number of Clusters": k_range, "Inertia": inertias})

    # Membuat plot interaktif dengan Plotly
    fig = px.line(
        df_elbow,
        x="Number of Clusters",
        y="Inertia",
        markers=True,
        title="Elbow Method For Optimal k",
        labels={"Inertia": "Sum of squared distances (Inertia)"},
    )

    # Menampilkan plot
    st.plotly_chart(fig)

    st.sidebar.subheader("Nilai Jumlah K")
    clust = st.sidebar.slider("Pilih jumlah kluster:", 2, 10, 3, 1)

    def k_means(n_clust):
        kmeans = KMeans(n_clusters=n_clust)

        y_cluster21 = kmeans.fit_predict(x_train21)
        data21["Cluster"] = y_cluster21
        colors = [plt.cm.nipy_spectral(float(i) / n_clust) for i in range(n_clust)]

        # Membuat dataframe baru dengan informasi yang diperlukan
        df_plotly = pd.DataFrame(
            {
                "Total KBG Perempuan": data21["Total KBG Perempuan"],
                "Provinsi": data21["Provinsi"],
                "Cluster": data21["Cluster"],
                "Komnas Perempuan": data21["Komnas Perempuan"],
                "Data Lembaga Mitra": data21["Data Lembaga Mitra"],
                "Badilag": data21["Badilag"],
            }
        )

        # Membuat scatter plot interaktif dengan Plotly
        fig = px.scatter(
            df_plotly,
            x="Total KBG Perempuan",
            y="Provinsi",
            color="Cluster",
            color_discrete_sequence=colors,
            labels={"Cluster": "Cluster"},
            title="Persebaran Kejahatan Berbasis Gender Terhadap Perempuan",
            hover_name="Provinsi",
            hover_data={
                "Komnas Perempuan": True,
                "Data Lembaga Mitra": True,
                "Badilag": True,
                "Total KBG Perempuan": True,
                "Provinsi": False,
                "Cluster": True,
            },
        )

        # Menampilkan plot
        st.plotly_chart(fig)
        st.write(data21)

    k_means(clust)


pages = {
    "-": intro,
    "2021": tahun2021,
    "2022": tahun2022,
}
page = st.sidebar.selectbox("Pilih tahun", pages.keys())
pages[page]()


# def k_means(n_clust):
#     kmeans = KMeans(n_clusters=n_clust)
#     y_cluster21 = kmeans.fit_predict(x_train21)
#     data21["Cluster"] = y_cluster21
#     colors = [plt.cm.nipy_spectral(float(i) / n_clust) for i in range(n_clust)]
#     plt.figure(figsize=(10, 5))
#     for i in range(len(colors)):
#         plt.scatter(
#             x_train21[kmeans.labels_ == i, 0],
#             data21["Provinsi"][kmeans.labels_ == i],
#             c=colors[i],
#             label=f"Cluster {i}",
#         )
#     # Menambahkan legenda
#     plt.legend()
#     # Menampilkan plot
#     plt.show()
#     st.header("Cluster Plot")
#     st.pyplot(plt)
#     st.write(data21)


# k_means(clust)
def k_means(n_clust, train):
    kmeans = KMeans(n_clusters=n_clust)

    y_cluster = kmeans.fit_predict(train)
    data["Cluster"] = y_cluster
    colors = [plt.cm.nipy_spectral(float(i) / n_clust) for i in range(n_clust)]

    # Membuat dataframe baru dengan informasi yang diperlukan
    df_plotly = pd.DataFrame(
        {
            "Total KBG Perempuan": data["Total KBG Perempuan"],
            "Provinsi": data["Provinsi"],
            "Cluster": data["Cluster"],
            "Komnas Perempuan": data["Komnas Perempuan"],
            "Data Lembaga Mitra": data["Data Lembaga Mitra"],
            "Badilag": data["Badilag"],
        }
    )

    # Membuat scatter plot interaktif dengan Plotly
    fig = px.scatter(
        df_plotly,
        x="Total KBG Perempuan",
        y="Provinsi",
        color="Cluster",
        color_discrete_sequence=colors,
        labels={"Cluster": "Cluster"},
        title="Persebaran Kejahatan Berbasis Gender Terhadap Perempuan",
        hover_name="Provinsi",
        hover_data={
            "Komnas Perempuan": True,
            "Data Lembaga Mitra": True,
            "Badilag": True,
            "Total KBG Perempuan": True,
            "Provinsi": False,
            "Cluster": True,
        },
    )

    # Menampilkan plot
    st.plotly_chart(fig)
    st.write(data)


# def k_means(n_clust):
#     kmeans = KMeans(n_clusters=n_clust)
#     y_cluster22 = kmeans.fit_predict(x_train22)
#     data22["Cluster"] = y_cluster22
#     colors = [plt.cm.nipy_spectral(float(i) / n_clust) for i in range(n_clust)]
#     plt.figure(figsize=(10, 5))
#     for i in range(len(colors)):
#         plt.scatter(
#             x_train22[kmeans.labels_ == i, 0],
#             data22["Provinsi"][kmeans.labels_ == i],
#             c=colors[i],
#             label=f"Cluster {i}",
#         )
#     # Menambahkan legenda
#     plt.legend()
#     # Menampilkan plot
#     plt.show()
#     st.header("Cluster Plot")
#     st.pyplot(plt)
#     st.write(data22)

# elbow
# inertias = []
#     k_range = range(1, 10)
#     for k in k_range:
#         km = KMeans(n_clusters=k).fit(x_train22)
#         inertias.append(km.inertia_)

#     st.header("Elbow Method")
#     fig, ax = plt.subplots()
#     ax.plot(k_range, inertias, marker="o")
#     ax.set(
#         xlabel="Number of clusters (k)",
#         ylabel="Sum of squared distances (Inertia)",
#         title="Elbow Method",
#     )
#     ax.grid(True)
#     st.pyplot(fig)
