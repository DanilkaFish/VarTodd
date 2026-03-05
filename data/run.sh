#!/bin/bash

# Проходим по всем файлам, соответствующим паттерну
for file in gf2*_mult_fr_*.matrix.npy; do
    # Проверяем, существует ли файл (на случай если нет совпадений)
    [ -e "$file" ] || continue
    
    # Создаем новое имя: удаляем _mult_fr и .matrix
    newname=$(echo "$file" | sed 's/_mult_fr//;s/\.matrix//')
    
    # Переименовываем файл
    mv -v "$file" "$newname"
done

echo "Переименование завершено!"