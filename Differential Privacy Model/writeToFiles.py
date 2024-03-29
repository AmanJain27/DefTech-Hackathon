import csv
import os


def most_optimal_dataset(true, false):
    k = 0
    for i in range(len(true)):
        if true[i] == false[i]:
            k += 1

    return (k / len(true)) * 100


def update_contents_in_files(path, start_depth, end_depth, step, epsilons, column_names, x_g, y_g, survival):

    datasets_acc = []

    for d in range(start_depth, end_depth, step):
        d_c = 0
        folder_name = f"{path}"
        depth = f"Depth={d}"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
            os.chdir(folder_name)
        else:
            os.chdir(folder_name)
        if not os.path.isdir(depth):
            os.mkdir(depth)
        else:
            for e in range(len(epsilons)):

                filename = f"{folder_name}/Depth={d}/haberman_generalized{e+1}.csv"
                with open(filename, "w", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)
                    for i in range(x_g.shape[0]):
                        writer.writerow(
                            [list(x_g[column_names[0]])[i], list(x_g[column_names[1]])[i], list(x_g[column_names[2]])[i],
                             survival[d_c][e][i]])

                datasets_acc.append(most_optimal_dataset(
                    list(y_g), survival[d_c][e]))

        d_c += 1

    return datasets_acc
