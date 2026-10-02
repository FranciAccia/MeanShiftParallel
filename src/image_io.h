#pragma once

#include <string>

#include "mean_shift.h"

// Carica un'immagine (PNG/JPG/...) come dataset di pixel RGB, in entrambi i layout AoS e SoA.
// Restituisce false se il file non esiste o non è leggibile.
bool load_image(const std::string& path, Dataset& data);

// Salva l'immagine segmentata: ogni pixel prende il colore del centro del suo cluster.
bool save_segmentation(const std::string& path, const Dataset& data, const Result& result);
