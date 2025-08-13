#ifndef CUSTOMAPH
#define CUSTOMAPH

#include "accel_includes_noglib.h"

// structures for Map key
typedef struct
{
    int harmtosum;
    int harm;
} MapKey;

typedef struct
{
    float startr;
    int harmtosum;
    int harm;
} StartrHarmTuple;

// structures for Map value
typedef struct
{
    // startr batch
    double *startr_array;
    double *lastr_array;
    // shi
    subharminfo *shi;
    // StartrHarmTuple idx batch
    StartrHarmTuple *tuple_array;
    int count;
} MapValue;

// structures for Map Entry
typedef struct
{
    MapKey key;
    MapValue value;
    ffdotpows_cu *subharmonics_batch; // Compute the array pointer for the resulting batch, and store it here for convenient release later
} MapEntry;

// Compare whether the two keys are equal
int compareKeys(MapKey *key1, MapKey *key2)
{
    return key1->harmtosum == key2->harmtosum &&
           key1->harm == key2->harm;
}

// search map
MapEntry *getMap(MapEntry *map, int map_size, MapKey key)
{
    for (int i = 0; i < map_size; i++)
    {
        if (compareKeys(&key, &map[i].key))
        {
            return &map[i];
        }
    }
    return; // If no matching key is found, return NULL
}

void freeMap(MapEntry *map, int *map_size)
{
    for (int i = 0; i < *map_size; i++)
    {
        free(map[i].value.startr_array);
        free(map[i].value.lastr_array);
        free(map[i].value.tuple_array);
    }
}

// insert or update map
void insertOrUpdateMap(MapEntry *map, int *map_size, MapKey key, float startr, float lastr, subharminfo *shi, int harmtosum, int harm, int max_map_size)
{
    // Search the map for a matching key
    for (int i = 0; i < *map_size; i++)
    {
        if (compareKeys(&key, &map[i].key))
        {
            // if a matching key is found, update the entry
            map[i].value.startr_array[map[i].value.count] = startr;
            map[i].value.lastr_array[map[i].value.count] = lastr;
            map[i].value.tuple_array[map[i].value.count] = (StartrHarmTuple){startr, harmtosum, harm};
            map[i].value.count += 1;
            return;
        }
    }
    // if no matching key is found, add a new entry
    if (*map_size < max_map_size)
    {
        map[*map_size].key = key;
        map[*map_size].value.startr_array[0] = startr;
        map[*map_size].value.lastr_array[0] = lastr;
        map[*map_size].value.shi = shi;
        map[*map_size].value.tuple_array[0] = (StartrHarmTuple){startr, harmtosum, harm};
        map[*map_size].value.count += 1;
        (*map_size)++;
    }
    else
    {
        // handle the case of map overflow
        printf("Map overflow\n");
    }
}

#endif