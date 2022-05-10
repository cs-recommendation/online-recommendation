import os
import numpy as np
import scipy.io
from six.moves import xrange

class DataHelper:
    def __init__(self,data_dir):
        self.dblp_data_fold = data_dir
        self.paper_list = []
        self.author_list = []
        self.conf_list = []
        self.term_list = []

    def load_data(self):
        with open(os.path.join(self.dblp_data_fold, 'paper_author.txt')) as pa_file:
            pa_lines = pa_file.readlines()
        for line in pa_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.author_list.append(token[1])
        with open(os.path.join(self.dblp_data_fold, 'paper_conf.txt')) as pc_file:
            pc_lines = pc_file.readlines()
        for line in pc_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.conf_list.append(token[1])
        with open(os.path.join(self.dblp_data_fold, 'paper_term.txt')) as pt_file:
            pt_lines = pt_file.readlines()
        for line in pt_lines:
            token = line.strip('\n').split('\t')
            self.paper_list.append(token[0])
            self.term_list.append(token[1])
        self.paper_list = list(set(self.paper_list))
        self.author_list = list(set(self.author_list))
        self.conf_list = list(set(self.conf_list))
        self.term_list = list(set(self.term_list))
        print ('#paper:{}, #author:{}, #conf:{}, term:{}'.format(len(self.paper_list), len(self.author_list),
                                                                 len(self.conf_list), len(self.term_list)))
        pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in pa_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pa_adj_matrix[row][col] = 1

        pc_adj_matrix = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in pc_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pc_adj_matrix[row][col] = 1

        pt_adj_matrix = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        for line in pt_lines:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            pt_adj_matrix[row][col] = 1

        ap_adj_matrix = np.transpose(pa_adj_matrix)
        ac_adj_matrix = np.matmul(np.transpose(pa_adj_matrix), pc_adj_matrix)
        at_adj_matrix = np.matmul(ap_adj_matrix, pt_adj_matrix)
        apa_adj_matrix = np.matmul(ap_adj_matrix,ap_adj_matrix.transpose())
        aca_adj_matrix = np.matmul(ac_adj_matrix,ac_adj_matrix.transpose())
        ata_adj_matrix = np.matmul(at_adj_matrix,at_adj_matrix.transpose())
        pcp_adj_matrix = np.matmul(pc_adj_matrix,pc_adj_matrix.transpose())
        ptp_adj_matrix = np.matmul(pt_adj_matrix,pt_adj_matrix.transpose())
        pap_adj_matrix = np.matmul(pa_adj_matrix,pa_adj_matrix.transpose())

        print('save matrix...')

        self.save_adj(ac_adj_matrix,'apc')
        self.save_adj(at_adj_matrix,'apt')
        self.save_adj(apa_adj_matrix,'apa')
        self.save_adj(aca_adj_matrix,'apcpa')
        self.save_adj(ata_adj_matrix,'aptpa')

    def save_mat(self,matrix,relation_name):
        scipy.io.savemat(os.path.join(self.dblp_data_fold,relation_name),
                         {relation_name:matrix})

    def save_adj(self,matrix,relation_name):
        row, col = np.nonzero(matrix)
        with open(os.path.join(self.dblp_data_fold,relation_name+'.txt'),'w') as adj_file:
            for i in xrange(len(row)):
                adj_file.write(str(row[i])+'\t'+str(col[i])+'\t'+str(matrix[row[i]][col[i]])+'\n')


if __name__ == '__main__':
    dh = DataHelper('../data/dblp/')
    dh.load_data()

