//
// Created by sm108 on 2/1/21.
//

#ifndef GRB_FUSION_EXEC_INFO_H
#define GRB_FUSION_EXEC_INFO_H

#if defined(HAVE_EXEC_INFO_SUPPORT)
void print_block_info(struct exec_info *info, unsigned full) {
    printf(";%lu;%lu;%lu;%lu;%lu;%lu;", info->nnzA, info->nrowsA, info->ncolsA, info->nnzB, info->nrowsB, info->ncolsB);

    printf("%lf", info->entries[0].time);
    for (int i = 0; i < info->num_entries; i++) {
        int j = i - 1;
        while (info->entries[j].depth != info->entries[i].depth - 1 && j > 0) { j--; }
        if (info->entries[j].depth == info->entries[i].depth - 1) {
            info->entries[j].children_time += info->entries[i].time;
        }
    }

    for (int i = 0; i < info->num_entries; i++) {
        if (i + 1 >=  info->num_entries || info->entries[i].depth >= info->entries[i + 1].depth) {
            printf(";%s:%d;%lf", info->entries[i].name, info->entries[i].line, info->entries[i].time);
        }
    }
    printf("\n");

    if (full) {
        for (int i = 0; i < info->num_entries; i++) {
            for (int j = 0; j < info->entries[i].depth; j++) { printf("  "); }
            printf("%s:%-4d %s [%s] %lf %lf\n", info->entries[i].file, info->entries[i].line, info->entries[i].name,
                   info->entries[i].txt ? info->entries[i].txt : "", info->entries[i].time,
                   info->entries[i].children_time);
        }
        printf("\n");
    }


}
#endif

#endif //GRB_FUSION_EXEC_INFO_H
