import pandas as pd
import numpy as np
import anndata as ad
from scipy.spatial import ConvexHull
from typing import Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
import gzip
import io
import matplotlib.pyplot as plt
import random
from itertools import product
from shapely.geometry import Polygon
import geopandas as gpd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform, RandomLinkSplit
from torch_geometric.nn import radius_graph

def uint32_to_str(cell_id_uint32: int, dataset_suffix: str) -> str:
    """
    Convert a 32-bit unsigned integer cell ID to a string with a specific suffix.

    Parameters:
    cell_id_uint32 (int): The 32-bit unsigned integer cell ID.
    dataset_suffix (str): The suffix to append to the string representation of the cell ID.

    Returns:
    str: The string representation of the cell ID with the appended suffix.
    """
    hex_prefix = hex(cell_id_uint32)[2:].zfill(8)
    hex_to_str_mapping = {
        '0': 'a', '1': 'b', '2': 'c', '3': 'd',
        '4': 'e', '5': 'f', '6': 'g', '7': 'h',
        '8': 'i', '9': 'j', 'a': 'k', 'b': 'l',
        'c': 'm', 'd': 'n', 'e': 'o', 'f': 'p'
    }
    str_prefix = ''.join([hex_to_str_mapping[char] for char in hex_prefix])
    return f"{str_prefix}-{dataset_suffix}"


def filter_transcripts(transcripts_df: pd.DataFrame, min_qv: float = 20.0) -> pd.DataFrame:
    """
    Filters transcripts based on quality value and removes unwanted transcripts.

    Parameters:
    transcripts_df (pd.DataFrame): The dataframe containing transcript data.
    min_qv (float): The minimum quality value threshold for filtering transcripts.

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    return transcripts_df[
        (transcripts_df['qv'] >= min_qv) &
        (~transcripts_df["feature_name"].str.startswith("NegControlProbe_")) &
        (~transcripts_df["feature_name"].str.startswith("antisense_")) &
        (~transcripts_df["feature_name"].str.startswith("NegControlCodeword_")) &
        (~transcripts_df["feature_name"].str.startswith("BLANK_")) & 
        (~transcripts_df["feature_name"].str.startswith("DeprecatedCodeword_"))
    ]


def compute_transcript_metrics(df: pd.DataFrame, qv_threshold: float = 30, cell_id_col: str = 'cell_id') -> Dict[str, Any]:
    """
    Computes various metrics for a given dataframe of transcript data filtered by quality value threshold.

    Parameters:
    df (pd.DataFrame): The dataframe containing transcript data.
    qv_threshold (float): The quality value threshold for filtering transcripts.
    cell_id_col (str): The name of the column representing the cell ID.

    Returns:
    Dict[str, Any]: A dictionary containing:
        - 'percent_assigned' (float): The percentage of assigned transcripts.
        - 'percent_cytoplasmic' (float): The percentage of cytoplasmic transcripts among assigned transcripts.
        - 'percent_nucleus' (float): The percentage of nucleus transcripts among assigned transcripts.
        - 'percent_non_assigned_cytoplasmic' (float): The percentage of non-assigned cytoplasmic transcripts among all non-assigned transcripts.
        - 'gene_metrics' (pd.DataFrame): A dataframe containing gene-level metrics:
            - 'feature_name': The gene name.
            - 'percent_assigned': The percentage of assigned transcripts for each gene.
            - 'percent_cytoplasmic': The percentage of cytoplasmic transcripts for each gene.
    """
    df_filtered = df[df['qv'] > qv_threshold]
    total_transcripts = len(df_filtered)
    assigned_transcripts = df_filtered[df_filtered[cell_id_col] != -1]
    percent_assigned = len(assigned_transcripts) / total_transcripts * 100
    cytoplasmic_transcripts = assigned_transcripts[assigned_transcripts['overlaps_nucleus'] != 1]
    percent_cytoplasmic = len(cytoplasmic_transcripts) / len(assigned_transcripts) * 100
    percent_nucleus = 100 - percent_cytoplasmic
    non_assigned_transcripts = df_filtered[df_filtered[cell_id_col] == -1]
    non_assigned_cytoplasmic = non_assigned_transcripts[non_assigned_transcripts['overlaps_nucleus'] != 1]
    percent_non_assigned_cytoplasmic = len(non_assigned_cytoplasmic) / len(non_assigned_transcripts) * 100

    gene_group_assigned = assigned_transcripts.groupby('feature_name')
    gene_group_all = df_filtered.groupby('feature_name')
    gene_percent_assigned = (gene_group_assigned.size() / gene_group_all.size() * 100).reset_index(name='percent_assigned')
    cytoplasmic_gene_group = cytoplasmic_transcripts.groupby('feature_name')
    gene_percent_cytoplasmic = (cytoplasmic_gene_group.size() / len(cytoplasmic_transcripts) * 100).reset_index(name='percent_cytoplasmic')
    gene_metrics = pd.merge(gene_percent_assigned, gene_percent_cytoplasmic, on='feature_name', how='outer').fillna(0)

    results = {
        'percent_assigned': percent_assigned,
        'percent_cytoplasmic': percent_cytoplasmic,
        'percent_nucleus': percent_nucleus,
        'percent_non_assigned_cytoplasmic': percent_non_assigned_cytoplasmic,
        'gene_metrics': gene_metrics
    }
    return results


class BuildTxGraph(BaseTransform):
    def __init__(self, r: float, loop: bool = False, max_num_neighbors: int = 32, flow: str = 'source_to_target', num_workers: int = 5) -> None:
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

    def forward(self, data: HeteroData) -> HeteroData:
        assert data['tx'].pos is not None
        data['tx', 'neighbors', 'tx'].edge_index = radius_graph(
            data['tx'].pos, self.r, max_num_neighbors=self.max_num_neighbors, num_workers=self.num_workers
        )
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'


class XeniumSample:
    def __init__(self, transcripts_df: pd.DataFrame = None, transcripts_radius: int = 10, nuclei_graph: bool = False):
        self.transcripts_df = transcripts_df
        self.transcripts_radius = transcripts_radius
        self.nuclei_graph = nuclei_graph
        if self.transcripts_df is not None:
            self.x_max = self.transcripts_df['x_location'].max()
            self.y_max = self.transcripts_df['y_location'].max()
            self.x_min = self.transcripts_df['x_location'].min()
            self.y_min = self.transcripts_df['y_location'].min()

    def crop_transcripts(self, x: float, y: float, x_size: int = 1000, y_size: int = 1000) -> 'XeniumSample':
        cropped_transcripts_df = self.transcripts_df[
            (self.transcripts_df['x_location'] > x) &
            (self.transcripts_df['x_location'] < x + x_size) &
            (self.transcripts_df['y_location'] > y) &
            (self.transcripts_df['y_location'] < y + y_size)
        ]
        return XeniumSample(cropped_transcripts_df)

    def load_transcripts(self, base_path: Path = None, sample: str = None, transcripts_filename: str = "transcripts.csv.gz", 
                         path: Path = None, min_qv: int = 20) -> 'XeniumSample':
        if path:
            with gzip.open(path, 'rb') as file:
                self.transcripts_df = pd.read_csv(io.BytesIO(file.read()))
                print(sample, len(self.transcripts_df))
        else:
            with gzip.open(base_path / sample / transcripts_filename, 'rb') as file:
                self.transcripts_df = pd.read_csv(io.BytesIO(file.read()))
                print(sample, len(self.transcripts_df))

        self.transcripts_df = filter_transcripts(self.transcripts_df, min_qv=min_qv)
        self.x_max = self.transcripts_df['x_location'].max()
        self.y_max = self.transcripts_df['y_location'].max()
        self.x_min = self.transcripts_df['x_location'].min()
        self.y_min = self.transcripts_df['x_location'].min()

        genes = self.transcripts_df[['feature_name']]
        self.tx_encoder = OneHotEncoder()
        self.tx_encoder.fit(genes)

        return self

    def load_nuclei(self, base_path: Path = None, sample: str = None, nuclei_filename: str = "nucleus_boundaries.csv.gz", 
                    path: Path = None) -> 'XeniumSample':
        if path:
            with gzip.open(path, 'rb') as file:
                self.nuclei_df = pd.read_csv(io.BytesIO(file.read()))
                print(sample, len(self.nuclei_df))
        else:
            with gzip.open(base_path / sample / nuclei_filename, 'rb') as file:
                self.nuclei_df = pd.read_csv(io.BytesIO(file.read()))
                print(sample, len(self.nuclei_df))
        return self

    @staticmethod
    def unassign_all_except_nucleus(transcripts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Unassign all transcripts except those overlapping with the nucleus.

        Parameters:
        transcripts_df (pd.DataFrame): The dataframe containing transcript data.

        Returns:
        pd.DataFrame: The dataframe with unassigned transcripts.
        """
        unassigned_df = transcripts_df.copy()
        unassigned_df.loc[unassigned_df['overlaps_nucleus'] == 0, 'cell_id'] = 'UNASSIGNED'
        return unassigned_df

    def save_dataset_for_segger(self, processed_dir: Path, d_x: int = 900, d_y: int = 900, x_size: int = 1000, y_size: int = 1000, 
                                margin_x: int = None, margin_y: int = None, compute_labels: bool = True, r: int = 5, 
                                val_prob: float = 0.1, test_prob: float = 0.2, neg_sampling_ratio_approx: int = 5, k_nc: int = 4, 
                                dist_nc: int = 20, k_tx: int = 5, dist_tx: int = 10, sampling_rate: float = 1, **kwargs) -> None:
        """
        Saves the dataset for Segger in a processed format.

        Parameters:
        processed_dir (Path): Directory to save the processed dataset.
        d_x (int): Step size in x direction for tiles.
        d_y (int): Step size in y direction for tiles.
        x_size (int): Width of each tile.
        y_size (int): Height of each tile.
        margin_x (int): Margin in x direction.
        margin_y (int): Margin in y direction.
        compute_labels (bool): Whether to compute labels.
        r (int): Radius for building the graph.
        val_prob (float): Probability of assigning a tile to the validation set.
        test_prob (float): Probability of assigning a tile to the test set.
        neg_sampling_ratio_approx (int): Approximate ratio of negative samples.
        k_nc (int): Number of nearest neighbors for nuclei.
        dist_nc (int): Distance threshold for nuclei.
        k_tx (int): Number of nearest neighbors for transcripts.
        dist_tx (int): Distance threshold for transcripts.
        sampling_rate (float): Rate of sampling tiles.
        """
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / 'train_tiles/processed').mkdir(parents=True, exist_ok=True)
        (processed_dir / 'test_tiles/processed').mkdir(parents=True, exist_ok=True)
        (processed_dir / 'val_tiles/processed').mkdir(parents=True, exist_ok=True)

        if margin_x is None:
            margin_x = d_x // 10
        if margin_y is None:
            margin_y = d_y // 10

        x_range = np.arange(self.x_min // 1000 * 1000, self.x_max, d_x)
        y_range = np.arange(self.y_min // 1000 * 1000, self.y_max, d_y)

        x_masks_nc = [(self.nuclei_df['vertex_x'] > x) & (self.nuclei_df['vertex_x'] < x + x_size) for x in x_range]
        y_masks_nc = [(self.nuclei_df['vertex_y'] > y) & (self.nuclei_df['vertex_y'] < y + y_size) for y in y_range]
        x_masks_tx = [(self.transcripts_df['x_location'] > (x - 10)) & (self.transcripts_df['x_location'] < (x + x_size + 10)) for x in x_range]
        y_masks_tx = [(self.transcripts_df['y_location'] > (y - 10)) & (self.transcripts_df['y_location'] < (y + y_size + 10)) for y in y_range]

        for i, j in tqdm(product(range(len(x_range)), range(len(y_range)))):
            prob = random.random()
            if prob > sampling_rate:
                continue
            x_mask = x_masks_nc[i]
            y_mask = y_masks_nc[j]
            mask = x_mask & y_mask
            if mask.sum() == 0:
                continue
            nc_df = self.nuclei_df[mask]
            nc_df = self.nuclei_df.loc[self.nuclei_df.cell_id.isin(nc_df.cell_id), :]
            x_mask_tx = x_masks_tx[i]
            y_mask_tx = y_masks_tx[j]
            tx_mask = x_mask_tx & y_mask_tx
            transcripts_df = self.transcripts_df[tx_mask]
            if transcripts_df.shape[0] == 0 or nc_df.shape[0] == 0:
                continue
            data = self.build_pyg_data_from_tile(nc_df, transcripts_df, compute_labels=compute_labels, **kwargs)
            data = BuildTxGraph(r=r)(data)
            try:
                if compute_labels:
                    transform = RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, edge_types=[('tx', 'belongs', 'nc')], 
                                                neg_sampling_ratio=neg_sampling_ratio_approx * 2)
                    data, _, _ = transform(data)
                    edge_index = data[('tx', 'belongs', 'nc')].edge_index
                    if edge_index.shape[1] < 10:
                        continue
                    edge_label_index = data[('tx', 'belongs', 'nc')].edge_label_index
                    data[('tx', 'belongs', 'nc')].edge_label_index = edge_label_index[:, torch.nonzero(
                        torch.any(edge_label_index[0].unsqueeze(1) == edge_index[0].unsqueeze(0), dim=1)
                    ).squeeze()]
                    data[('tx', 'belongs', 'nc')].edge_label = data[('tx', 'belongs', 'nc')].edge_label[torch.nonzero(
                        torch.any(edge_label_index[0].unsqueeze(1) == edge_index[0].unsqueeze(0), dim=1)
                    ).squeeze()]
                coords_nc = data['nc'].pos
                coords_tx = data['tx'].pos[:, :2]
                data['tx'].tx_field = self.get_edge_index(coords_tx, coords_tx, k=k_tx, dist=dist_tx, type='kd_tree')
                data['tx'].nc_field = self.get_edge_index(coords_nc, coords_tx, k=k_nc, dist=dist_nc, type='kd_tree')
                filename = f"tiles_x{int(x_size)}_y{int(y_size)}_{i}_{j}.pt"
                prob = random.random()
                if prob > val_prob + test_prob:
                    torch.save(data, processed_dir / 'train_tiles/processed' / filename)
                elif prob > val_prob:
                    torch.save(data, processed_dir / 'test_tiles/processed' / filename)
                else:
                    torch.save(data, processed_dir / 'val_tiles/processed' / filename)
            except:
                pass

    @staticmethod
    def compute_nuclei_geometries(nuclei_df: pd.DataFrame, area: bool = True, convexity: bool = True, elongation: bool = True, 
                                  circularity: bool = True) -> gpd.GeoDataFrame:
        """
        Computes geometries for nuclei from the dataframe.

        Parameters:
        nuclei_df (pd.DataFrame): The dataframe containing nuclei data.
        area (bool): Whether to compute area.
        convexity (bool): Whether to compute convexity.
        elongation (bool): Whether to compute elongation.
        circularity (bool): Whether to compute circularity.

        Returns:
        gpd.GeoDataFrame: A geodataframe containing computed geometries.
        """
        grouped = nuclei_df.groupby('cell_id')
        polygons = [Polygon(list(zip(group_data['vertex_x'], group_data['vertex_y']))) for _, group_data in grouped]
        polygons = gpd.GeoSeries(polygons)
        gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
        gdf['cell_id'] = list(grouped.groups.keys())
        gdf[['x', 'y']] = gdf.centroid.get_coordinates()
        if area:
            gdf['area'] = polygons.area
        if convexity:
            gdf['convexity'] = polygons.convex_hull.area / polygons.area
        if elongation:
            r = polygons.minimum_rotated_rectangle
            gdf['elongation'] = r.area / (r.length * r.width)
        if circularity:
            r = gdf.minimum_bounding_radius()
            gdf['circularity'] = polygons.area / (r * r)
        return gdf

    @staticmethod
    def get_edge_index(coords_1: np.ndarray, coords_2: np.ndarray, k: int = 100, dist: int = 10, type: str = 'edge_index') -> torch.Tensor:
        """
        Computes edge indices using KDTree for given coordinates.

        Parameters:
        coords_1 (np.ndarray): First set of coordinates.
        coords_2 (np.ndarray): Second set of coordinates.
        k (int): Number of nearest neighbors.
        dist (int): Distance threshold.
        type (str): Type of edge index to return.

        Returns:
        torch.Tensor: Edge indices.
        """
        tree = KDTree(coords_1)
        d_kdtree, idx_out = tree.query(coords_2, k=k, distance_upper_bound=dist)
        if type == 'kd_tree':
            return torch.tensor(idx_out, dtype=torch.long)
        edge_index = XeniumSample.kd_to_edge_index_((idx_out.shape[0], coords_1.shape[0]), idx_out)
        if type == 'edge_index':
            return edge_index
        if type == 'adj':
            return torch_geometric.utils.to_dense_adj(edge_index)[0].to_sparse()

    @staticmethod
    def kd_to_edge_index_(shape: tuple, idx_out: np.ndarray) -> torch.Tensor:
        """
        Converts KDTree indices to edge indices.

        Parameters:
        shape (tuple): Shape of the original matrix.
        idx_out (np.ndarray): Indices from KDTree query.

        Returns:
        torch.Tensor: Edge indices.
        """
        idx = np.column_stack((np.arange(shape[0]), idx_out))
        idc = pd.DataFrame(idx).melt(0).loc[lambda df: df['value'] != shape[1], :]
        edge_index = torch.tensor(idc[['variable', 'value']].to_numpy(), dtype=torch.long).t().contiguous()
        return edge_index

    def build_pyg_data_from_tile(self, nuclei_df: pd.DataFrame, transcripts_df: pd.DataFrame, compute_labels: bool = True, 
                                 **kwargs) -> HeteroData:
        """
        Builds PyG data from a tile of nuclei and transcripts data.

        Parameters:
        nuclei_df (pd.DataFrame): Dataframe containing nuclei data.
        transcripts_df (pd.DataFrame): Dataframe containing transcripts data.
        compute_labels (bool): Whether to compute labels.
        **kwargs: Additional arguments for geometry computation.

        Returns:
        HeteroData: PyG Heterogeneous Data object.
        """
        data = HeteroData()
        nc_args = list(inspect.signature(self.compute_nuclei_geometries).parameters)
        nc_dict = {k: kwargs.pop(k) for k in kwargs if k in nc_args}
        nc_gdf = self.compute_nuclei_geometries(nuclei_df, **nc_dict)
        max_area = nc_gdf['area'].max()
        tx_args = list(inspect.signature(self.get_edge_index).parameters)
        tx_dict = {k: kwargs.pop(k) for k in kwargs if k in tx_args}
        x_xyz = torch.as_tensor(transcripts_df[['x_location', 'y_location', 'z_location']].values).float()
        data['tx'].id = transcripts_df[['transcript_id']].values
        data['tx'].pos = x_xyz
        x_features = torch.as_tensor(self.tx_encoder.transform(transcripts_df[['feature_name']]).toarray()).float()
        data['tx'].x = x_features.to_sparse()
        tx_edge_index = self.get_edge_index(nc_gdf[['x', 'y']].values, transcripts_df[['x_location', 'y_location']].values, 
                                            k=3, dist=np.sqrt(max_area) * 10)
        data['tx', 'neighbors', 'nc'].edge_index = tx_edge_index
        ind = np.where((transcripts_df.overlaps_nucleus == 1) & (transcripts_df.cell_id.isin(nc_gdf['cell_id'])))[0]
        tx_nc_edge_index = np.column_stack((ind, np.searchsorted(nc_gdf['cell_id'].values, transcripts_df.iloc[ind]['cell_id'].values)))
        data['nc'].id = nc_gdf[['cell_id']].values
        data['nc'].pos = nc_gdf[['x', 'y']].values
        nc_x = nc_gdf.iloc[:, 4:]
        data['nc'].x = torch.as_tensor(nc_x.to_numpy()).float()
        data['tx', 'belongs', 'nc'].edge_index = torch.as_tensor(tx_nc_edge_index.T).long()
        return data



