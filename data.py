from torch_geometric.data import Data, download_url, extract_gz
import networkx as nx

disgenetGeneURL = "https://ctdbase.org/reports/CTD_genes_diseases.tsv.gz"

extract_gz(download_url(disgenetGeneURL, 'data'), 'data')



