import csv


def report_exp_results(report_file_name, exp_names_list, metrics_list, times_list):
    assert len(metrics_list) == len(times_list)
    assert report_file_name.endswith('.csv')

    data = []
    for exp_name, metrics, time_elapsed in zip(exp_names_list, metrics_list, times_list):
        data.append({
            'exp_name': exp_name,
            'accuracy_micro': metrics.accuracy_micro,
            'accuracy_macro': metrics.accuracy_macro,
            'f1_macro': metrics.f1_macro,
            'time_elapsed': time_elapsed,
        })

    print(f'Started reporting exp esults to {report_file_name}')
    with open(report_file_name, 'w', newline='') as csvfile:
        fieldnames = ['exp_name', 'accuracy_micro', 'accuracy_macro', 'f1_macro', 'time_elapsed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f'Finished reporting exp results to {report_file_name}')
