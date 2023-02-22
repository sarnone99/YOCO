import numpy as np
from sklearn.cluster import AgglomerativeClustering


''' NORMAL '''
def npAggloCluster(y1,y2,y3,num_classes):

    agglo = AgglomerativeClustering(linkage='average', n_clusters=num_classes, affinity='cosine')

    # for each batch
    y1_batch_mean_embeddings = np.zeros((y1.shape[0], y1.shape[1], y1.shape[2], num_classes), dtype=np.float32)
    y2_batch_mean_embeddings = np.zeros((y2.shape[0], y2.shape[1], y2.shape[2], num_classes), dtype=np.float32)
    y3_batch_mean_embeddings = np.zeros((y3.shape[0], y3.shape[1], y3.shape[2], num_classes), dtype=np.float32)
    for i in range(y1.shape[0]):
        feats1 = y1[i,:]
        feats1 = feats1.reshape(feats1.shape[0]*feats1.shape[1], feats1.shape[2]).T
        feats2 = y2[i,:]
        feats2 = feats2.reshape(feats2.shape[0]*feats2.shape[1], feats2.shape[2]).T
        feats3 = y3[i,:]
        feats3 = feats3.reshape(feats3.shape[0]*feats3.shape[1], feats3.shape[2]).T

        feats1 = np.nan_to_num(feats1)
        feats2 = np.nan_to_num(feats2)
        feats3 = np.nan_to_num(feats3)
        
        y1_clustered = agglo.fit_predict(feats1)
        y2_clustered = agglo.fit_predict(feats2)
        y3_clustered = agglo.fit_predict(feats3)
        
        # For each class
        y1_mean_embeddings = []
        y2_mean_embeddings = []
        y3_mean_embeddings = []
        for c in range(num_classes):
            mean_embedding1 = np.mean(feats1[:][y1_clustered == c], axis=0)
            y1_mean_embeddings.append(mean_embedding1)
            mean_embedding2 = np.mean(feats2[:][y2_clustered == c], axis=0)
            y2_mean_embeddings.append(mean_embedding2)
            mean_embedding3 = np.mean(feats3[:][y3_clustered == c], axis=0)
            y3_mean_embeddings.append(mean_embedding3)
            
        y1_mean_embeddings = np.array(y1_mean_embeddings)
        y1_mean_embeddings = np.stack(y1_mean_embeddings)
        y2_mean_embeddings = np.array(y2_mean_embeddings)
        y2_mean_embeddings = np.stack(y2_mean_embeddings)
        y3_mean_embeddings = np.array(y3_mean_embeddings)
        y3_mean_embeddings = np.stack(y3_mean_embeddings)
    
        y1_mean_embeddings = y1_mean_embeddings.reshape(num_classes,y1.shape[1],y1.shape[2]).T
        y1_batch_mean_embeddings[i] = y1_mean_embeddings
        y2_mean_embeddings = y2_mean_embeddings.reshape(num_classes,y2.shape[1],y2.shape[2]).T
        y2_batch_mean_embeddings[i] = y2_mean_embeddings
        y3_mean_embeddings = y3_mean_embeddings.reshape(num_classes,y3.shape[1],y3.shape[2]).T
        y3_batch_mean_embeddings[i] = y3_mean_embeddings

    return [y1_batch_mean_embeddings, y2_batch_mean_embeddings, y3_batch_mean_embeddings]


class YOLO_Kmeans:
    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = "2012_train.txt"
        
    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2012_train.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
