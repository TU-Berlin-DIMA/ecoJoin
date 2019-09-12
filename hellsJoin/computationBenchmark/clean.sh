sort -k1 tm > tm_sort
sed -i -r 's/\S+//1' tm_sort
sed -i -r 's/\S+//1' tm_sort
sed -i -r 's/\S+//2' tm_sort
sed -i -r 's/\S+//2' tm_sort
sed -i -r 's/\S+//2' tm_sort
uniq tm_sort > tm_sort_u
wc -l tm_sort_u
