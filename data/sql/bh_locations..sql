SELECT wm.WellNumber AS well_name,
	wm.api_wellno,
	wm.bh_ftns,
	bh.bh_lat,
	bh.bh_long
FROM [tblWellFieldPool] w
INNER JOIN [tblRefGeoPool] p
ON w.FieldPoolCode = p.FieldPoolCode
LEFT JOIN [tblWellMaster] wm 
ON w.API_WellNo = wm.API_WellNo
LEFT JOIN [tblBH_LatLong] bh
ON w.api_wellno = bh.API_WellNo 
WHERE p.PoolName = 'PRUDHOE BAY, PRUDHOE OIL'
	AND wm.bh_ns IS NOT null 

