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
import plotly.graph_objects as go


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

        cluster_totals = (
            df_plotly.groupby("Cluster")["Total KBG Perempuan"].sum().reset_index()
        )
        total_provinsi = (
            df_plotly.groupby("Cluster")["Provinsi"]
            .count()
            .reset_index(name="Total Provinsi")
        )

        # Bar plot untuk menunjukkan berapa banyak data di masing-masing klaster (vertical)
        fig_bar = px.bar(
            df_plotly,
            x="Cluster",
            y="Total KBG Perempuan",  # Menggunakan 'y' sebagai sumbu vertikal
            orientation="v",  # Mengatur orientasi ke vertical
            title="Jumlah Data di Setiap Klaster",
            labels={"Cluster": "Klaster", "Total KBG Perempuan": "Jumlah Data"},
            hover_data=["Provinsi"],
            color="Provinsi",  # Memberikan warna berbeda pada setiap klaster
            # color_discrete_sequence=colors,  # Menggunakan palet warna bawaan Plotly
        )

        # Menambahkan total jumlah data pada masing-masing klaster
        for i, total, total_prov in zip(
            cluster_totals["Cluster"],
            cluster_totals["Total KBG Perempuan"],
            total_provinsi["Total Provinsi"],
        ):
            fig_bar.add_annotation(
                x=i,
                y=total,
                text=f"Total KBG Perempuan: {total}<br>Total Provinsi: {total_prov}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30,
            )

        st.plotly_chart(fig_bar)

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

        cluster_totals = (
            df_plotly.groupby("Cluster")["Total KBG Perempuan"].sum().reset_index()
        )
        total_provinsi = (
            df_plotly.groupby("Cluster")["Provinsi"]
            .count()
            .reset_index(name="Total Provinsi")
        )

        # Bar plot untuk menunjukkan berapa banyak data di masing-masing klaster (vertical)
        fig_bar = px.bar(
            df_plotly,
            x="Cluster",
            y="Total KBG Perempuan",  # Menggunakan 'y' sebagai sumbu vertikal
            orientation="v",  # Mengatur orientasi ke vertical
            title="Jumlah Data di Setiap Klaster",
            labels={"Cluster": "Klaster", "Total KBG Perempuan": "Jumlah Data"},
            hover_data=["Provinsi"],
            color="Provinsi",  # Memberikan warna berbeda pada setiap klaster
            # color_discrete_sequence=colors,  # Menggunakan palet warna bawaan Plotly
        )

        # Menambahkan total jumlah data pada masing-masing klaster
        for i, total, total_prov in zip(
            cluster_totals["Cluster"],
            cluster_totals["Total KBG Perempuan"],
            total_provinsi["Total Provinsi"],
        ):
            fig_bar.add_annotation(
                x=i,
                y=total,
                text=f"Total KBG Perempuan: {total}<br>Total Provinsi: {total_prov}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30,
            )

        st.plotly_chart(fig_bar)
        # # Bar plot untuk menunjukkan berapa banyak data di masing-masing klaster (vertical)
        # fig_bar = px.bar(
        #     df_plotly,
        #     x="Cluster",
        #     y="Total KBG Perempuan",  # Menggunakan 'y' sebagai sumbu vertikal
        #     orientation="v",  # Mengatur orientasi ke vertical
        #     title="Jumlah Data di Setiap Klaster",
        #     labels={"Cluster": "Klaster", "Total KBG Perempuan": "Jumlah Data"},
        #     hover_data=["Provinsi"],
        #     color="Provinsi",  # Memberikan warna berbeda pada setiap klaster
        #     # color_discrete_sequence=colors,  # Menggunakan palet warna bawaan Plotly
        # )

        # st.plotly_chart(fig_bar)

    k_means(clust)


def train_and_predict(year, x_train, input_data, clust):
    # Scaling data training
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # Melatih model KMeans
    kmeans = KMeans(n_clusters=clust)
    kmeans.fit(x_train_scaled)

    # Scaling data input using the same scaler
    input_data_scaled = scaler.transform(np.array([sum(input_data)]).reshape(1, -1))

    # Lakukan prediksi klaster
    predicted_cluster = kmeans.predict(input_data_scaled)

    # Dapatkan label klaster untuk setiap data di dataset
    cluster_labels = kmeans.labels_

    return predicted_cluster[0], cluster_labels


def show_scatter_plot(dataset, predicted_cluster, input_data, nama_provinsi, clust):
    # Membuat dataframe baru dengan informasi yang diperlukan
    df_plotly = pd.DataFrame(
        {
            "Total KBG Perempuan": dataset["Total KBG Perempuan"],
            "Provinsi": dataset["Provinsi"],
            "Cluster": dataset["Cluster"],
            "Komnas Perempuan": dataset["Komnas Perempuan"],
            "Data Lembaga Mitra": dataset["Data Lembaga Mitra"],
            "Badilag": dataset["Badilag"],
        }
    )

    # Membuat scatter plot interaktif dengan Plotly
    fig = px.scatter(
        df_plotly,
        x="Total KBG Perempuan",
        y="Provinsi",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set1,
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

    # Menambahkan titik prediksi ke dalam plot
    fig.add_trace(
        go.Scatter(
            x=[sum(input_data)],
            y=[nama_provinsi],
            mode="markers",
            marker=dict(color="red"),
            name="Prediksi",
            hoverinfo="text",  # Mengganti "name+x+y" menjadi "text"
            # text=f"Total KBG Perempuan: {total}<br>Total Provinsi: {total_prov}",
            text=[
                f"<b>{nama_provinsi}</b><br><br>Total KBG Perempuan: {sum(input_data)}<br>Komnas Perempuan: {input_data[0]}<br>Data Lembaga Mitra: {input_data[1]}<br>Badilag: {input_data[2]}<br>Cluster: {predicted_cluster}"
            ],
            showlegend=True,
        )
    )

    # Menampilkan plot
    st.plotly_chart(fig)
    # Tambahkan data hasil prediksi ke dalam DataFrame df_plotly
    df_prediksi = pd.DataFrame(
        {
            "Cluster": [predicted_cluster],
            "Total KBG Perempuan": [sum(input_data)],
            "Provinsi": [nama_provinsi],
        }
    )
    df_plotly = pd.concat([df_plotly, df_prediksi], ignore_index=True)

    # Bar plot untuk menunjukkan berapa banyak data di masing-masing klaster (vertical)
    fig_bar = px.bar(
        df_plotly,
        x="Cluster",
        y="Total KBG Perempuan",
        orientation="v",
        title="Jumlah Data di Setiap Klaster",
        labels={"Cluster": "Klaster", "Total KBG Perempuan": "Jumlah Data"},
        hover_data=["Provinsi"],  # Menambahkan Provinsi pada hover
        color="Provinsi",  # Warna merah untuk data prediksi
        color_discrete_map={"Prediksi": "red"},
    )

    # Menampilkan bar plot vertical
    st.plotly_chart(fig_bar)


def barplot_prediksi(n_clust, df_plotly, input_data, predicted_cluster):
    # Tambahkan data hasil prediksi ke dalam DataFrame df_plotly
    df_prediksi = pd.DataFrame(
        {
            "Cluster": [predicted_cluster],
            "Total KBG Perempuan": [sum(input_data)],
            "Provinsi": ["Prediksi"],
        }
    )
    df_plotly = pd.concat([df_plotly, df_prediksi], ignore_index=True)

    # Bar plot untuk menunjukkan berapa banyak data di masing-masing klaster (vertical)
    fig_bar = px.bar(
        df_plotly,
        x="Cluster",
        y="Total KBG Perempuan",
        orientation="v",
        title="Jumlah Data di Setiap Klaster",
        labels={"Cluster": "Klaster", "Total KBG Perempuan": "Jumlah Data"},
        hover_data=["Provinsi"],  # Menambahkan Provinsi pada hover
    )

    # Menampilkan bar plot vertical
    st.plotly_chart(fig_bar)


def prediksi():
    year = st.selectbox("Pilih tahun untuk prediksi", ["2021", "2022"])

    # Membaca dataset
    dataset = pd.read_excel("dataset_kbg_perempuan.xlsx", sheet_name=year)

    # Menambahkan kolom Total KBG Perempuan
    dataset["Total KBG Perempuan"] = (
        dataset["Komnas Perempuan"] + dataset["Data Lembaga Mitra"] + dataset["Badilag"]
    )

    # Menampilkan dataset
    st.header("Dataset")
    st.write(dataset)

    # Mempersiapkan data untuk training
    x_train = dataset[["Total KBG Perempuan"]].values

    # Masukkan data prediksi
    nama_provinsi = st.text_input("Masukkan Nama Provinsi:")
    komnas_perempuan = st.number_input("Jumlah KBG Perempuan dari Komnas Perempuan:", 0)
    data_lembaga_mitra = st.number_input(
        "Jumlah KBG Perempuan dari Data Lembaga Mitra:", 0
    )
    badilag = st.number_input("Jumlah KBG Perempuan dari Badilag:", 0)

    # Pilihan jumlah kluster
    clust = st.slider("Pilih jumlah kluster:", 2, 10, 3, 1, key="kluster")

    # Button untuk melakukan prediksi
    if st.button("Prediksi Klaster"):
        input_data = [komnas_perempuan, data_lembaga_mitra, badilag]
        predicted_cluster, cluster_labels = train_and_predict(
            year, x_train, input_data, clust
        )

        # Tambahkan kolom 'Cluster' ke dalam DataFrame dataset
        dataset["Cluster"] = cluster_labels

        st.success(
            f"Provinsi {nama_provinsi} pada tahun {year} diprediksi masuk ke dalam Klaster {predicted_cluster}"
        )

        # Menampilkan scatter plot
        show_scatter_plot(dataset, predicted_cluster, input_data, nama_provinsi, clust)
        # barplot_prediksi(clust, df_plotly, input_data, predicted_cluster)
    # st.header("Prediksi Klaster")

    # # Pilih tahun untuk prediksi
    # tahun_prediksi = st.selectbox("Pilih tahun untuk prediksi", ["2021", "2022"])

    # # Masukkan data prediksi
    # nama_provinsi = st.text_input("Masukkan Nama Provinsi:")
    # komnas_perempuan = st.number_input("Jumlah KBG Perempuan dari Komnas Perempuan:", 0)
    # data_lembaga_mitra = st.number_input(
    #     "Jumlah KBG Perempuan dari Data Lembaga Mitra:", 0
    # )
    # badilag = st.number_input("Jumlah KBG Perempuan dari Badilag:", 0)
    # clust = st.slider("Pilih jumlah kluster:", 2, 10, 3, 1)


pages = {
    "-": intro,
    "2021": tahun2021,
    "2022": tahun2022,
    "Prediksi": prediksi,
}
st.sidebar.header("Menu")
page = st.sidebar.selectbox("Pilih Menu", pages.keys())
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
